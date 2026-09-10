#!/usr/bin/env python3
"""
Verify UnityRenderBridge against a stubbed transport.

The bridge's job is bookkeeping around the simulation: export once, throttle
publishing, and notice when the simulator has compiled a new scene. None of that
needs a real peer connection, so the transport is stubbed here and the WebRTC
path is left to scripts/test_unity_webrtc.py.

The check that matters is the episode boundary. SIMPLE resamples objects per
episode and compiles a fresh MjModel when it does; if the bridge misses that,
Unity keeps the old geometry and the stream drives the wrong objects -- body 7
being a mug one episode and a drill the next, with nothing in the packets to say
so.

Usage
-----
  python scripts/test_unity_bridge.py [--mjcf path/to/scene.xml]

Needs only mujoco and numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import mujoco

from simple.teleop.unity.bridge import UnityRenderBridge

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


class StubServer:
    """Records what the bridge hands it, in place of a peer connection."""

    def __init__(self, scene_id: int) -> None:
        self.scene_id = scene_id
        self.connected = True
        self.published = []
        self.closed = False

    def set_scene_id(self, scene_id: int) -> None:
        self.scene_id = scene_id

    def publish(self, frame, positions, quaternions) -> bool:
        if not self.connected:
            return False
        self.published.append((frame, self.scene_id, len(positions)))
        return True

    def stats(self) -> dict:
        return {"sent": len(self.published), "open": self.connected}

    def close(self) -> None:
        self.closed = True


class StubSimulator:
    """Duck-types the parts of MujocoSimulator the bridge reads."""

    def __init__(self, model) -> None:
        self.set_model(model)
        self.render_step = 0

    def set_model(self, model) -> None:
        self.mjModel = model
        self.mjData = mujoco.MjData(model)
        mujoco.mj_forward(self.mjModel, self.mjData)

    def step(self) -> None:
        self.mjData.qpos[: self.mjModel.nq] += 0.001
        mujoco.mj_forward(self.mjModel, self.mjData)
        self.render_step += 1


def build_variant(mjcf_path: str):
    """Compile the scene with one extra body, standing in for a new episode."""
    spec = mujoco.MjSpec.from_file(mjcf_path)
    body = spec.worldbody.add_body(name="episode_prop", pos=[0.5, 0.0, 0.5])
    body.add_geom(
        name="episode_prop_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.05, 0.05, 0.05],
    )
    return spec.compile()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mjcf", default=DEFAULT_MJCF)
    args = parser.parse_args()

    if not os.path.isfile(args.mjcf):
        print(f"MJCF nao encontrado: {args.mjcf}", file=sys.stderr)
        return 2

    model = mujoco.MjSpec.from_file(args.mjcf).compile()
    sim = StubSimulator(model)

    with tempfile.TemporaryDirectory(prefix="simple_unity_bridge_") as out_dir:
        try:
            print("\n[1] Export na construcao")
            stub = StubServer(scene_id=0)
            # publish_hz=0 disables throttling, so each tick is one packet and
            # the counts below do not depend on how fast this loop runs.
            bridge = UnityRenderBridge(sim, out_dir=out_dir, publish_hz=0, server=stub)
            check(
                "scene.json escrito",
                os.path.isfile(os.path.join(out_dir, "scene.json")),
            )
            check("um unico export", bridge.exports == 1, f"exports={bridge.exports}")
            check(
                "scene_id nao trivial", bridge.scene_id != 0, f"0x{bridge.scene_id:08x}"
            )

            print("\n[2] Publicacao e throttling")
            for _ in range(5):
                sim.step()
                bridge.tick()
            check(
                "sem throttle, um pacote por tick",
                len(stub.published) == 5,
                f"{len(stub.published)} pacotes",
            )
            check(
                "contagem de bodies correta",
                all(n == model.nbody for _, _, n in stub.published),
                f"{model.nbody} bodies",
            )
            check(
                "frame vem do render_step do simulador",
                [f for f, _, _ in stub.published] == [1, 2, 3, 4, 5],
                str([f for f, _, _ in stub.published]),
            )

            slow = StubServer(scene_id=0)
            slow_bridge = UnityRenderBridge(
                sim, out_dir=out_dir, publish_hz=2.0, server=slow
            )
            for _ in range(20):
                sim.step()
                slow_bridge.tick()
            check(
                "throttle a 2 Hz segura os pacotes",
                len(slow.published) <= 2,
                f"{len(slow.published)} pacotes em 20 ticks",
            )
            check("force=True ignora o throttle", slow_bridge.tick(force=True) is True)

            print("\n[3] Desconectado")
            stub.connected = False
            before = len(stub.published)
            sim.step()
            check("tick desconectado retorna False", bridge.tick() is False)
            check("nada publicado", len(stub.published) == before)
            stub.connected = True

            print("\n[4] Troca de cena entre episodios")
            old_scene_id = bridge.scene_id
            with open(os.path.join(out_dir, "scene.json"), encoding="utf-8") as fh:
                old_bodies = len(json.load(fh)["bodies"])

            sim.set_model(build_variant(args.mjcf))  # o que reset() faria

            check("resync detecta o modelo novo", bridge.resync() is True)
            check("reexportou", bridge.exports == 2, f"exports={bridge.exports}")
            check(
                "scene_id mudou",
                bridge.scene_id != old_scene_id,
                f"0x{old_scene_id:08x} -> 0x{bridge.scene_id:08x}",
            )
            check(
                "servidor adotou o scene_id novo",
                stub.scene_id == bridge.scene_id,
                f"0x{stub.scene_id:08x}",
            )

            with open(os.path.join(out_dir, "scene.json"), encoding="utf-8") as fh:
                new_bodies = len(json.load(fh)["bodies"])
            check(
                "scene.json em disco tem a geometria nova",
                new_bodies == old_bodies + 1,
                f"{old_bodies} -> {new_bodies} bodies",
            )

            check("resync sem mudanca e no-op", bridge.resync() is False)
            check("sem export extra", bridge.exports == 2)

            time.sleep(0.01)
            sim.step()
            bridge.tick(force=True)
            check(
                "pacotes seguintes carregam o scene_id novo",
                stub.published[-1][1] == bridge.scene_id,
            )

            print("\n[5] Simulador sem cena compilada")

            class Unbuilt:
                render_step = 0  # mjModel only exists after update_layout()

            try:
                UnityRenderBridge(Unbuilt(), out_dir=out_dir, server=StubServer(0))
                message = ""
            except RuntimeError as exc:
                message = str(exc)
            check(
                "erro explica que falta o env.reset()",
                "reset()" in message,
                message[:60] or "(nenhuma excecao)",
            )

            print("\n[6] Encerramento")
            bridge.close()
            check("servidor injetado nao e fechado pela ponte", stub.closed is False)

            owned = UnityRenderBridge(sim, out_dir=out_dir, server=StubServer(0))
            owned.close()
            check("stats reporta exports e scene_id", "exports" in owned.stats())
        except Failure as exc:
            print(f"\nFALHOU: {exc}")
            return 1

    print("\nTodos os testes passaram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
