"""
Stream scene state to Unity over an unreliable WebRTC data channel.

Pose state is disposable: every frame supersedes the one before it, so a packet
that goes missing costs nothing as long as the next one arrives on time. An
ordered channel gets that backwards -- one lost packet stalls everything behind
it while SCTP retransmits, and the operator sees the scene freeze and then jump.
So the channel is always **unordered**.

Reliability is a separate axis, and the first version got it wrong. It also set
``maxRetransmits=0``, reasoning that a dropped frame is free. That holds only
while a message fits inside one MTU. Above it SCTP fragments, and an
unreliable fragmented message survives only if every fragment does -- against a
libwebrtc peer across a 1280-byte Tailscale link, none did: the channel opened,
both ends agreed on the settings, and nothing above the MTU ever arrived. A
20-byte probe landed; a 1395-byte one did not.

The default is therefore unordered and reliable. Head-of-line blocking, the
thing worth avoiding, comes from ordering rather than from retransmission: an
unordered channel lets later frames overtake a message still being retransmitted,
and the receiver drops the straggler as stale when it finally lands. Pass
``max_retransmits=0`` to get the old behaviour, which is a real win once packets
fit one MTU -- see ``SAFE_PAYLOAD_BYTES``.

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

import numpy as np

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


def ensure_console_logging() -> None:
    """Make this package's INFO output visible on the console.

    SIMPLE configures no logging at all, so the effective level is WARNING and
    every connection message here goes nowhere. That is survivable right up
    until a headset will not connect, at which point the log this code already
    writes -- signaling up, peer state, channel opened -- is exactly what says
    where the handshake stopped, and none of it reaches the operator.

    A library has no business reconfiguring the root logger, so the handler is
    attached to ``simple.teleop.unity`` only and does nothing if the
    application has already set one up.
    """
    package_logger = logging.getLogger("simple.teleop.unity")
    if package_logger.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[unity] %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)


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


def _require_webrtc_dependencies() -> None:
    """Fail at startup if the optional WebRTC dependencies are missing.

    aiortc is only needed when --unity is passed, so it lives in the ``unity``
    extra and is imported lazily. Importing it inside the connection handler,
    though, defers the failure to the worst possible moment: the server binds
    the port, reports itself ready, and only dies when a client arrives --
    reaching that client as close code 1011 with the cause on neither side.

    Checking here means a missing install is reported once, at startup, by the
    process that can still do something about it.
    """
    missing = []
    for module, package in (
        ("aiortc", "aiortc>=1.9"),
        ("websockets", "websockets>=12.0"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        raise RuntimeError(
            "Unity state streaming needs "
            + " and ".join(missing)
            + ", which are not installed. They sit in the optional 'unity' "
            "extra:\n"
            "    uv pip install " + " ".join(f'"{p}"' for p in missing) + "\n"
            "or install the whole extra with:\n"
            "    uv pip install -e '.[unity]'"
        )


def rewrite_host_candidates(sdp: str, host: str) -> str:
    """Replace the address of every host candidate in ``sdp`` with ``host``.

    On a machine with several interfaces -- a physical NIC, a Tailscale tun,
    maybe a docker bridge -- aiortc advertises a host candidate for each, and
    the headset may spend its connection attempt on one that is unreachable
    from its side of the tailnet. Pinning the address it should use avoids the
    guesswork.
    """
    if not host:
        return sdp, 0
    lines = []
    rewritten = 0
    for line in sdp.splitlines():
        if line.startswith("a=candidate:") and " typ host " in line:
            match = _CANDIDATE_HOST_RE.match(line)
            if match:
                line = f"{match.group(1)}{host}{match.group(3)}"
                rewritten += 1
        lines.append(line)
    return "\r\n".join(lines), rewritten


# Attributes some builds of Unity's WebRTC refuse. Current aiortc emits
# neither, but python_webrtc.py filtered them and the cost of keeping the
# filter is nil.
_UNITY_REJECTED_SDP_ATTRS = ("a=extmap-allow-mixed", "a=ice-options")


def clean_sdp_for_unity(sdp: str) -> str:
    """Shape an answer SDP so Unity's WebRTC will accept it.

    Unity parses the answer by splitting on line breaks and rejoining with
    CRLF, then appending one more CRLF of its own (``WebRTCUtils.ApplySDP``).
    aiortc's SDP already ends in CRLF, so that split yields a trailing empty
    element and the rejoin lands a blank line at the end -- which libwebrtc
    rejects outright as "Invalid SDP line", naming no line in particular.

    Returning without a trailing newline makes that round trip lossless. It is
    the same shape python_webrtc.py sends, which is the version known to work
    against this client.
    """
    cleaned = [
        line
        for line in sdp.splitlines()
        if line.strip() and not line.startswith(_UNITY_REJECTED_SDP_ATTRS)
    ]
    return "\r\n".join(cleaned)


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
        max_retransmits: int | None = None,
    ) -> None:
        self._scene_id = scene_id
        self._loop = loop
        self._buffer_limit = buffer_limit
        self._lock = threading.Lock()
        self.sent = 0
        self.dropped_backpressure = 0
        self.dropped_closed = 0

        self._warned_fragmentation = False

        self._channel = pc.createDataChannel(
            label, ordered=False, maxRetransmits=max_retransmits
        )

        @self._channel.on("open")
        def _on_open():
            logger.info(
                "state channel open (ordered=%s, maxRetransmits=%s)",
                self._channel.ordered,
                self._channel.maxRetransmits,
            )
            self._send_size_probe()

        @self._channel.on("close")
        def _on_close():
            logger.info("state channel closed after %d frames", self.sent)

    def _send_size_probe(self) -> None:
        """Send one small and one full-size message when the channel opens.

        Both carry the magic of a state packet but a body count of zero, so a
        receiver decodes them as a valid, empty frame and counts them without
        moving anything.

        The pair separates two failures that look identical from the far end,
        where nothing arrives and no counter moves. Read the receiver's counters
        after connecting:

            applied 1, malformed 1  both sizes arrive; size is not the problem
            applied 1, malformed 0  only the small one; fragmentation is
            applied 0, malformed 0  nothing arrives; the channel is not
                                    delivering at all, and size is a red herring

        The small probe decodes as a valid zero-body frame, so it lands in
        ``applied``. The padded one declares zero bodies but carries a longer
        body, so the length check rejects it into ``droppedMalformed`` -- which
        is what makes its arrival visible without moving any geometry.
        """
        empty = np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        small = encode_state(0, self._scene_id, *empty)
        # Padding rides in an oversized frame declaring zero bodies, so the
        # receiver rejects it cleanly rather than trying to place phantom links.
        padded = small + b"\x00" * (SAFE_PAYLOAD_BYTES + 200 - len(small))

        logger.info(
            "size probe: sending %d bytes then %d bytes "
            "(if only the first is received, fragmentation is the problem)",
            len(small),
            len(padded),
        )
        for packet in (small, padded):
            try:
                self._channel.send(packet)
            except Exception:
                logger.warning("size probe send failed", exc_info=True)

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
            if self._channel.maxRetransmits == 0:
                logger.warning(
                    "state packet is %d bytes for %d bodies, above the ~%d that "
                    "fits one 1280-byte MTU, and this channel does not "
                    "retransmit. A fragmented message survives only if every "
                    "fragment does; against a libwebrtc peer none did. Drop "
                    "max_retransmits=0 or get the packet under the MTU.",
                    len(packet),
                    len(positions),
                    SAFE_PAYLOAD_BYTES,
                )
            else:
                logger.info(
                    "state packet is %d bytes for %d bodies, above the ~%d that "
                    "fits one 1280-byte MTU, so SCTP fragments it. The channel "
                    "retransmits, so this costs latency on loss rather than the "
                    "frame itself.",
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
        verbose: bool = True,
    ) -> None:
        if verbose:
            ensure_console_logging()
        _require_webrtc_dependencies()
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

        logger.info("websockets %s", getattr(websockets, "__version__", "unknown"))
        async with websockets.serve(self._signaling_entry, self._host, self._port):
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

    async def _signaling_entry(self, websocket, path=None) -> None:
        """Entry point websockets calls, wrapping the session.

        ``path`` exists only for compatibility. Up to websockets 13 the server
        called handlers as ``handler(websocket, path)``; from 14 it passes the
        connection alone. This package allows websockets >= 12, so a handler
        taking one argument raises TypeError under the older releases.

        The try/except is what makes that kind of failure findable. websockets
        reports a crashed handler to the client as a bare close code 1011 and
        logs the cause on its own logger, which nothing here configures -- so
        the session dies with the reason visible on neither side.
        """
        try:
            await self._handle_signaling(websocket)
        except Exception:
            logger.exception("signaling handler crashed")
            raise

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
        logger.info("signaling: client connected")

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
                    logger.info("signaling: offer received")
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
                        sdp, rewritten = rewrite_host_candidates(sdp, self._ice_host)
                        logger.info(
                            "pinned %d host candidate(s) to %s",
                            rewritten,
                            self._ice_host,
                        )
                    sdp = clean_sdp_for_unity(sdp)

                    await websocket.send(
                        json.dumps({"type": pc.localDescription.type, "sdp": sdp})
                    )
                    logger.info("signaling: answer sent, waiting for the peer")

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
