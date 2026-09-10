"""
Stream scene state to Unity over an unreliable WebRTC data channel.

Pose state is disposable: every frame supersedes the one before it, so a packet
that goes missing costs nothing as long as the next one arrives on time. A
reliable ordered channel gets that exactly backwards -- one lost packet stalls
the stream behind retransmissions and the operator sees the scene freeze and
then jump. The state channel is therefore opened with ``ordered=False`` and
``maxRetransmits=0``: a lost packet is simply gone, and the next frame lands on
schedule.

Who opens the channel
---------------------
Unity is the offerer and opens its own reliable ``tracker`` channel for poses
and controller state. The state channel runs the other way, and is opened here
by the answering peer after the offer is applied -- WebRTC allows either side to
open a channel once the SCTP association exists. That keeps the reliability
settings in one place instead of split across two languages, and the settings do
reach the far side: a peer receiving this channel sees ``ordered == false`` and
``maxRetransmits == 0`` on its own handler.

On the Unity side that means handling ``pc.OnDataChannel`` rather than creating
the channel:

    pc.OnDataChannel = ch => {
        if (ch.Label == "state") { ch.OnMessage = OnStateMessage; }
    };

Two entry points
----------------
``UnityStateChannel`` attaches to an existing ``RTCPeerConnection``, so it can be
dropped into a signaling server that already exists. ``UnityStateServer`` is a
self-contained signaling server for the SIMPLE side.

``aiortc`` is an optional dependency; install with the ``unity`` extra.
"""

import asyncio
import json
import logging
import re
import threading

from .protocol import encode_state

logger = logging.getLogger(__name__)

STATE_CHANNEL_LABEL = "state"
TRACKER_CHANNEL_LABEL = "tracker"

# Largest state packet that crosses a Tailscale link in one piece.
#
# Tailscale's tun MTU is 1280 bytes. Subtracting IP (20), UDP (8), the DTLS
# record with its GCM nonce and tag (~29), the SCTP common header (12) and the
# DATA chunk header (16) leaves roughly 1195 bytes of payload.
#
# This matters more than usual here because the channel is configured with no
# retransmits: SCTP delivers a fragmented message only if every fragment
# arrives, so a packet split into k pieces is lost with probability
# 1-(1-p)^k. Fragmenting quietly multiplies the drop rate the unreliable
# channel was chosen to keep low.
SAFE_PAYLOAD_BYTES = 1195

_CANDIDATE_HOST_RE = re.compile(r"^(a=candidate:[^ ]+ \d+ \w+ \d+ )([^ ]+)( .*)$")


def is_routable_candidate(address: str) -> bool:
    """False for candidates a remote peer can never reach.

    Link-local addresses are per-interface and meaningless across a link;
    advertising them just adds ICE pairs that are guaranteed to fail and delays
    the connection while they time out.
    """
    if not address:
        return False
    if address.startswith("169.254."):
        return False
    return not address.lower().startswith(("fe80:", "fc00:", "fd00:"))


def rewrite_host_candidates(sdp: str, host: str) -> str:
    """Replace the address of every host candidate in ``sdp`` with ``host``.

    On a machine with several interfaces -- a physical NIC, a Tailscale tun,
    maybe a docker bridge -- aiortc advertises a host candidate for each, and
    the headset may spend its connection attempt on one that is unreachable
    from its side of the tailnet. Pinning the address it should use avoids the
    guesswork.
    """
    if not host:
        return sdp
    lines = []
    for line in sdp.splitlines():
        if line.startswith("a=candidate:") and " typ host " in line:
            match = _CANDIDATE_HOST_RE.match(line)
            if match:
                line = f"{match.group(1)}{host}{match.group(3)}"
        lines.append(line)
    return "\r\n".join(lines) + "\r\n"


# Above this many bytes queued on the channel, the link is not keeping up.
# Dropping the current frame is the right response: the next one is a better
# picture of the world than a backlog of stale ones.
DEFAULT_BUFFER_LIMIT = 256 * 1024


class UnityStateChannel:
    """An unreliable state channel on an existing peer connection.

    Publishing is safe to call from the simulation thread; the send itself is
    marshalled onto the event loop thread that owns the connection.
    """

    def __init__(
        self,
        pc,
        scene_id: int,
        loop: asyncio.AbstractEventLoop,
        buffer_limit: int = DEFAULT_BUFFER_LIMIT,
        label: str = STATE_CHANNEL_LABEL,
    ) -> None:
        self._scene_id = scene_id
        self._loop = loop
        self._buffer_limit = buffer_limit
        self._lock = threading.Lock()
        self.sent = 0
        self.dropped_backpressure = 0
        self.dropped_closed = 0

        self._warned_fragmentation = False

        self._channel = pc.createDataChannel(label, ordered=False, maxRetransmits=0)

        @self._channel.on("open")
        def _on_open():
            logger.info(
                "state channel open (ordered=%s, maxRetransmits=%s)",
                self._channel.ordered,
                self._channel.maxRetransmits,
            )

        @self._channel.on("close")
        def _on_close():
            logger.info("state channel closed after %d frames", self.sent)

    @property
    def is_open(self) -> bool:
        return self._channel.readyState == "open"

    def set_scene_id(self, scene_id: int) -> None:
        """Point the channel at a new scene without renegotiating.

        The simulator can compile a new scene between episodes. Tearing down the
        peer connection to match would drop the headset's session, so the id
        carried by subsequent packets is swapped instead; a client that has not
        reloaded the geometry sees the mismatch and can ignore the stream.
        """
        with self._lock:
            self._scene_id = scene_id

    def publish(self, frame: int, positions, quaternions) -> bool:
        """Queue one frame of body poses. Returns True if it was handed off.

        Never raises on a closed or congested channel -- a dropped frame is a
        normal outcome here, not an error, and the simulation loop should not
        have to guard every call.
        """
        if not self.is_open:
            with self._lock:
                self.dropped_closed += 1
            return False

        if self._channel.bufferedAmount > self._buffer_limit:
            with self._lock:
                self.dropped_backpressure += 1
            return False

        with self._lock:
            scene_id = self._scene_id
        packet = encode_state(frame, scene_id, positions, quaternions)

        if len(packet) > SAFE_PAYLOAD_BYTES and not self._warned_fragmentation:
            self._warned_fragmentation = True
            logger.warning(
                "state packet is %d bytes for %d bodies, above the ~%d that fits "
                "one 1280-byte MTU (Tailscale, most VPNs). SCTP will fragment it, "
                "and with no retransmits a message survives only if every "
                "fragment does -- expect the drop rate to rise with body count.",
                len(packet),
                len(positions),
                SAFE_PAYLOAD_BYTES,
            )

        self._loop.call_soon_threadsafe(self._send, packet)
        with self._lock:
            self.sent += 1
        return True

    def _send(self, packet: bytes) -> None:
        # Runs on the event loop thread. The channel can close between the
        # readyState check in publish() and this callback.
        try:
            if self._channel.readyState == "open":
                self._channel.send(packet)
        except Exception:
            logger.warning("state channel send failed", exc_info=True)

    def stats(self) -> dict:
        with self._lock:
            return {
                "sent": self.sent,
                "dropped_backpressure": self.dropped_backpressure,
                "dropped_closed": self.dropped_closed,
                "open": self.is_open,
            }


class UnityStateServer:
    """WebSocket signaling plus a state channel, driven from a sync sim loop.

    Unity connects, sends an offer and its ICE candidates; this answers and
    opens the state channel back. The event loop runs on its own thread so the
    simulation loop can call ``publish`` without becoming async.

    Messages on Unity's own ``tracker`` channel are handed to ``on_tracker``,
    which is where wrist poses and controller state will arrive when the teleop
    agent is wired up.
    """

    def __init__(
        self,
        scene_id: int,
        host: str = "0.0.0.0",
        port: int = 8765,
        on_tracker=None,
        ice_servers=None,
        buffer_limit: int = DEFAULT_BUFFER_LIMIT,
        ice_host: str | None = None,
    ) -> None:
        self._scene_id = scene_id
        self._host = host
        self._port = port
        self._on_tracker = on_tracker
        self._ice_servers = ice_servers or ["stun:stun.l.google.com:19302"]
        self._buffer_limit = buffer_limit
        self._ice_host = ice_host

        self._state: UnityStateChannel | None = None
        self._pcs = set()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="unity-state-server", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    # -- lifecycle ---------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        import websockets

        async with websockets.serve(self._handle_signaling, self._host, self._port):
            logger.info("Unity signaling on ws://%s:%d", self._host, self._port)
            self._ready.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.1)

        for pc in list(self._pcs):
            await pc.close()
        self._pcs.clear()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # -- signaling ---------------------------------------------------------

    async def _handle_signaling(self, websocket) -> None:
        from aiortc import (
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
        )

        config = RTCConfiguration(iceServers=[RTCIceServer(urls=self._ice_servers)])
        pc = RTCPeerConnection(configuration=config)
        self._pcs.add(pc)
        pending_candidates = []
        remote_set = False

        @pc.on("connectionstatechange")
        async def _on_state():
            logger.info("peer connection: %s", pc.connectionState)

        @pc.on("datachannel")
        def _on_datachannel(channel):
            logger.info("Unity opened channel %r", channel.label)
            if channel.label != TRACKER_CHANNEL_LABEL or self._on_tracker is None:
                return

            @channel.on("message")
            def _on_message(message):
                try:
                    self._on_tracker(message)
                except Exception:
                    logger.warning("tracker callback raised", exc_info=True)

        try:
            async for raw in websocket:
                data = json.loads(raw)
                kind = data.get("type")

                if kind == "offer":
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=data["sdp"], type="offer")
                    )
                    remote_set = True
                    for candidate in pending_candidates:
                        await pc.addIceCandidate(candidate)
                    pending_candidates.clear()

                    # After the offer is applied the SCTP association exists, so
                    # the channel can be opened from this side.
                    self._state = UnityStateChannel(
                        pc,
                        scene_id=self._scene_id,
                        loop=self._loop,
                        buffer_limit=self._buffer_limit,
                    )

                    await pc.setLocalDescription(await pc.createAnswer())
                    while pc.iceGatheringState != "complete":
                        await asyncio.sleep(0.1)

                    sdp = pc.localDescription.sdp
                    if self._ice_host:
                        sdp = rewrite_host_candidates(sdp, self._ice_host)
                        logger.info("pinned host candidates to %s", self._ice_host)

                    await websocket.send(
                        json.dumps({"type": pc.localDescription.type, "sdp": sdp})
                    )

                elif kind == "candidate":
                    candidate = self._parse_candidate(data)
                    if candidate is None:
                        await pc.addIceCandidate(None)
                    elif not is_routable_candidate(candidate.ip):
                        logger.debug("ignoring unroutable candidate %s", candidate.ip)
                    elif remote_set:
                        await pc.addIceCandidate(candidate)
                    else:
                        pending_candidates.append(candidate)

                elif kind == "bye":
                    break
        except Exception:
            logger.warning("signaling session ended with an error", exc_info=True)
        finally:
            await pc.close()
            self._pcs.discard(pc)
            self._state = None

    @staticmethod
    def _parse_candidate(data):
        from aiortc.sdp import candidate_from_sdp

        raw = data.get("candidate")
        if not raw:
            return None
        candidate = candidate_from_sdp(
            raw.split(":", 1)[1] if raw.startswith("candidate:") else raw
        )
        candidate.sdpMid = data.get("sdpMid", "0")
        candidate.sdpMLineIndex = data.get("sdpMLineIndex", 0)
        return candidate

    # -- publishing --------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._state is not None and self._state.is_open

    def set_scene_id(self, scene_id: int) -> None:
        """Adopt a new scene id, for this connection and for any that follow."""
        self._scene_id = scene_id
        state = self._state
        if state is not None:
            state.set_scene_id(scene_id)

    def publish(self, frame: int, positions, quaternions) -> bool:
        """Publish one frame. No-op when Unity is not connected."""
        state = self._state
        if state is None:
            return False
        return state.publish(frame, positions, quaternions)

    def stats(self) -> dict:
        state = self._state
        return state.stats() if state is not None else {"open": False}
