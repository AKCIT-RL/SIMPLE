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
import threading

from .protocol import encode_state

logger = logging.getLogger(__name__)

STATE_CHANNEL_LABEL = "state"
TRACKER_CHANNEL_LABEL = "tracker"

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

        packet = encode_state(frame, self._scene_id, positions, quaternions)
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
    ) -> None:
        self._scene_id = scene_id
        self._host = host
        self._port = port
        self._on_tracker = on_tracker
        self._ice_servers = ice_servers or ["stun:stun.l.google.com:19302"]
        self._buffer_limit = buffer_limit

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
                    await websocket.send(
                        json.dumps(
                            {
                                "type": pc.localDescription.type,
                                "sdp": pc.localDescription.sdp,
                            }
                        )
                    )

                elif kind == "candidate":
                    candidate = self._parse_candidate(data)
                    if candidate is None:
                        await pc.addIceCandidate(None)
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

    def publish(self, frame: int, positions, quaternions) -> bool:
        """Publish one frame. No-op when Unity is not connected."""
        state = self._state
        if state is None:
            return False
        return state.publish(frame, positions, quaternions)

    def stats(self) -> dict:
        state = self._state
        return state.stats() if state is not None else {"open": False}
