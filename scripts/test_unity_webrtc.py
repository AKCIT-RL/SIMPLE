#!/usr/bin/env python3
"""
Loopback test for the Unity state channel over WebRTC.

Stands up ``UnityStateServer`` and a stand-in Unity client -- an aiortc offerer
that opens the reliable ``tracker`` channel exactly as ``WebRTCSignalingUnity``
does -- then checks:

  1. the state channel reaches the client with ordered=False, maxRetransmits=0
  2. real G1 poses survive the round trip byte for byte
  3. the tracker channel still carries Unity -> Python messages
  4. publishing while disconnected is a silent no-op, not an exception

The reliability settings are the point of the exercise: they are configured on
the answering side, and this is what proves they arrive intact on the other end
rather than being quietly reset to the defaults.

Usage
-----
  python scripts/test_unity_webrtc.py
  python scripts/test_unity_webrtc.py --mjcf path/to/scene.xml --frames 120

Needs the ``unity`` extra (aiortc, websockets) plus mujoco.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import mujoco
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription

from simple.teleop.unity import protocol
from simple.teleop.unity.scene_export import export_scene, world_poses
from simple.teleop.unity.webrtc_state import (
    SAFE_PAYLOAD_BYTES,
    UnityStateServer,
    clean_sdp_for_unity,
    is_routable_candidate,
    rewrite_host_candidates,
)

DEFAULT_MJCF = r"C:\Users\muril\Teleop6\Assets\Mujoco Unitree g1\scene.xml"


class Failure(Exception):
    pass


def check(name: str, condition: bool, detail: str = "") -> None:
    print(
        f"  [{'PASS' if condition else 'FAIL'}] {name}"
        + (f" -- {detail}" if detail else "")
    )
    if not condition:
        raise Failure(name)


class FakeUnityClient:
    """Minimal stand-in for WebRTCSignalingUnity: offers, opens `tracker`."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.pc = RTCPeerConnection()
        self.tracker = None
        self.state_channel_info = None
        self.received = []
        self._state_open = asyncio.Event()

    async def connect(self) -> None:
        @self.pc.on("datachannel")
        def _on_datachannel(channel):
            if channel.label != "state":
                return
            self.state_channel_info = {
                "label": channel.label,
                "ordered": channel.ordered,
                "maxRetransmits": channel.maxRetransmits,
            }

            @channel.on("message")
            def _on_message(message):
                self.received.append(message)

            self._state_open.set()

        self.tracker = self.pc.createDataChannel("tracker")

        async with websockets.connect(self.url) as ws:
            await self.pc.setLocalDescription(await self.pc.createOffer())
            while self.pc.iceGatheringState != "complete":
                await asyncio.sleep(0.05)
            await ws.send(
                json.dumps({"type": "offer", "sdp": self.pc.localDescription.sdp})
            )

            answer = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            await self.pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
            await asyncio.wait_for(self._state_open.wait(), timeout=10)
            # Hold the signaling socket open; the server tears the peer
            # connection down when it closes.
            await self._done.wait()

    async def run(self, body):
        self._done = asyncio.Event()
        task = asyncio.create_task(self.connect())
        try:
            await asyncio.wait_for(self._state_open.wait(), timeout=15)
            await body()
        finally:
            self._done.set()
            await asyncio.gather(task, return_exceptions=True)
            await self.pc.close()


def test_candidate_parsing() -> None:
    """Trickle ICE, which the loopback path below never exercises.

    The loopback client waits for gathering to finish and ships its candidates
    inside the SDP, so nothing here would otherwise touch ``_parse_candidate``.
    A real Unity client trickles them as separate signaling messages, and does
    it in whichever of these shapes its WebRTC implementation prefers.
    """
    print("\n[0] Parsing de ICE candidate (trickle)")
    cases = {
        "host com prefixo": "candidate:1 1 UDP 2130706431 192.168.1.10 54321 typ host",
        "host sem prefixo": "1 1 UDP 2130706431 192.168.1.10 54321 typ host",
        "srflx com raddr": (
            "candidate:2 1 UDP 1694498815 100.118.137.125 54322 typ srflx "
            "raddr 192.168.1.10 rport 54321"
        ),
    }
    for label, raw in cases.items():
        parsed = UnityStateServer._parse_candidate(
            {"candidate": raw, "sdpMid": "0", "sdpMLineIndex": 0}
        )
        check(
            label,
            parsed is not None and parsed.port in (54321, 54322),
            f"ip={parsed.ip}",
        )

    check(
        "fim de gathering (candidate nulo) vira None",
        UnityStateServer._parse_candidate({"candidate": None}) is None,
    )


def test_ice_helpers() -> None:
    """Multi-homed hosts and VPN links, which loopback cannot reproduce."""
    print("\n[0b] Ajustes de ICE para link Tailscale/VPN")

    # A multi-homed host as aiortc would describe it: LAN NIC, docker bridge,
    # and a reflexive candidate that must survive the rewrite untouched.
    sdp_lines = [
        "v=0",
        "a=candidate:1 1 udp 2130706431 192.168.1.10 5000 typ host generation 0",
        "a=candidate:2 1 udp 2130706431 172.17.0.1 5001 typ host generation 0",
        (
            "a=candidate:3 1 udp 1694498815 203.0.113.7 5002 typ srflx "
            "raddr 192.168.1.10 rport 5000"
        ),
        "a=mid:0",
    ]
    sdp = "\r\n".join(sdp_lines)
    out, rewritten = rewrite_host_candidates(sdp, "100.118.137.125")
    check(
        "todos os candidates host viram o IP do Tailscale",
        rewritten == 2 and out.count("100.118.137.125") == 2,
        f"{rewritten} de 2",
    )
    check("candidate srflx intocado", "203.0.113.7 5002 typ srflx" in out)
    check("linhas nao-candidate intocadas", "a=mid:0" in out and "v=0" in out)
    check("sem ice_host o SDP passa igual", rewrite_host_candidates(sdp, "")[0] == sdp)

    check("link-local IPv4 descartado", not is_routable_candidate("169.254.3.4"))
    check("link-local IPv6 descartado", not is_routable_candidate("fe80::1"))
    check("IP do Tailscale aceito", is_routable_candidate("100.118.137.125"))
    check("IP de LAN aceito", is_routable_candidate("192.168.1.10"))

    # 31 bodies fit one 1280-byte MTU; a scene with objects may not.
    fits = protocol.packet_size(31)
    check(
        "pacote do G1 cabe em uma MTU de VPN",
        fits <= SAFE_PAYLOAD_BYTES,
        f"{fits} <= {SAFE_PAYLOAD_BYTES} bytes",
    )
    limit = next(
        n for n in range(1, 500) if protocol.packet_size(n) > SAFE_PAYLOAD_BYTES
    )
    print(f"         fragmenta a partir de {limit} bodies")

    test_sdp_survives_unity_round_trip()


def unity_apply_sdp(sdp: str) -> str:
    """Reproduce what WebRTCSignalingUnity does to a received answer.

    It splits on either line ending with StringSplitOptions.None, rejoins with
    CRLF, and appends one more CRLF:

        p.sdp.Split(new[] { "\\r\\n", "\\n" }, StringSplitOptions.None)
        string.Join("\\r\\n", sdpLines) + "\\r\\n"
    """
    return "\r\n".join(re.split(r"\r\n|\n", sdp)) + "\r\n"


def test_sdp_survives_unity_round_trip() -> None:
    """The answer must not grow a blank line on the way into libwebrtc.

    aiortc ends its SDP with CRLF. Unity's split then yields a trailing empty
    element, the rejoin keeps it, and Unity's own appended CRLF turns it into a
    blank line -- which libwebrtc rejects as "Invalid SDP line" without saying
    which one. Stripping the trailing newline before sending is what keeps the
    round trip lossless.
    """
    print("\n[0c] SDP sobrevive ao parsing da Unity")

    raw = "v=0\r\na=mid:0\r\na=setup:active\r\n"  # como o aiortc entrega
    naive = unity_apply_sdp(raw)
    check(
        "SDP cru ganha linha em branco (o bug)",
        any(not line.strip() for line in naive.split("\r\n")[:-1]),
        "confirma o mecanismo",
    )

    cleaned = unity_apply_sdp(clean_sdp_for_unity(raw))
    blanks = [line for line in cleaned.split("\r\n")[:-1] if not line.strip()]
    check("SDP limpo nao ganha linha em branco", not blanks, f"{len(blanks)} em branco")
    check("conteudo preservado", "v=0" in cleaned and "a=setup:active" in cleaned)
    check(
        "sem newline no fim antes da Unity", not clean_sdp_for_unity(raw).endswith("\n")
    )

    messy = "v=0\r\n\r\na=ice-options:trickle\r\na=extmap-allow-mixed\r\na=mid:0\r\n"
    out = clean_sdp_for_unity(messy)
    check("linhas em branco removidas", "\r\n\r\n" not in out)
    check("a=ice-options removido", "ice-options" not in out)
    check("a=extmap-allow-mixed removido", "extmap-allow-mixed" not in out)


async def run_test(args) -> int:
    test_candidate_parsing()
    test_ice_helpers()

    model = mujoco.MjSpec.from_file(args.mjcf).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with tempfile.TemporaryDirectory(prefix="simple_unity_") as tmp:
        manifest = export_scene(model, tmp)
    scene_id = manifest["scene_id"]
    print(f"Modelo   : {model.nbody} bodies, scene_id 0x{scene_id:08x}")

    tracker_messages = []
    server = UnityStateServer(
        scene_id=scene_id,
        host="127.0.0.1",
        port=args.port,
        on_tracker=tracker_messages.append,
    )

    client = FakeUnityClient(f"ws://127.0.0.1:{args.port}")

    print("\n[1] Negociacao e configuracao do canal")

    async def body():
        info = client.state_channel_info
        check("canal 'state' recebido pela Unity", info is not None)
        check(
            "nao-ordenado (ordered=False)",
            info["ordered"] is False,
            f"ordered={info['ordered']}",
        )
        check(
            "sem retransmissao (maxRetransmits=0)",
            info["maxRetransmits"] == 0,
            f"maxRetransmits={info['maxRetransmits']}",
        )

        print("\n[2] Round trip de poses reais do G1")
        pos, quat = world_poses(model, data)
        sent = 0
        for frame in range(args.frames):
            # Nudge the model so consecutive frames differ.
            data.qpos[: model.nq] += 0.001
            mujoco.mj_forward(model, data)
            pos, quat = world_poses(model, data)
            if server.publish(frame, pos, quat):
                sent += 1
            await asyncio.sleep(0.005)

        for _ in range(100):
            if len(client.received) >= sent:
                break
            await asyncio.sleep(0.05)

        check(
            "pacotes entregues",
            len(client.received) == sent,
            f"{len(client.received)} de {sent} enviados",
        )

        decoded = [protocol.decode_state(p) for p in client.received]
        check(
            "todos com o scene_id correto",
            all(d["scene_id"] == scene_id for d in decoded),
        )
        check(
            "contagem de bodies correta",
            all(len(d["positions"]) == model.nbody for d in decoded),
            f"{model.nbody} bodies",
        )

        last = decoded[-1]
        check(
            "ultimo frame identico ao enviado",
            np.array_equal(last["positions"], pos)
            and np.array_equal(last["quaternions"], quat),
            f"frame {last['frame']}",
        )

        frames = [d["frame"] for d in decoded]
        check("nenhum frame duplicado", len(set(frames)) == len(frames))

        print("\n[3] Canal tracker (Unity -> Python)")
        client.tracker.send(json.dumps({"head": [1, 0, 0, 0]}))
        for _ in range(100):
            if tracker_messages:
                break
            await asyncio.sleep(0.05)
        check(
            "mensagem do tracker recebida",
            bool(tracker_messages),
            f"{len(tracker_messages)} msg",
        )

        stats = server.stats()
        print(f"\n         stats do servidor: {stats}")

    try:
        await client.run(body)
    except Failure as exc:
        print(f"\nFALHOU: {exc}")
        return 1
    finally:
        pass

    print("\n[4] Publicacao sem cliente conectado")
    await asyncio.sleep(0.5)  # let the server notice the peer went away
    try:
        result = server.publish(
            999, np.zeros((model.nbody, 3)), np.zeros((model.nbody, 4))
        )
        raised = False
    except Exception as exc:  # noqa: BLE001
        raised = True
        result = exc
    try:
        check("publish desconectado nao levanta excecao", not raised, str(result))
        check("publish desconectado retorna False", result is False)
    except Failure as exc:
        print(f"\nFALHOU: {exc}")
        return 1
    finally:
        server.close()

    print("\nTodos os testes passaram.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mjcf", default=DEFAULT_MJCF)
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    if not os.path.isfile(args.mjcf):
        print(f"MJCF nao encontrado: {args.mjcf}", file=sys.stderr)
        return 2

    return asyncio.run(run_test(args))


if __name__ == "__main__":
    raise SystemExit(main())
