"""
Drive a Unity headset from a running MuJoCo simulation.

``UnityRenderBridge`` exports the scene once, serves it over WebRTC, and pushes
body poses as the simulation advances. It is deliberately independent of which
agent is driving the robot: the Unity view is a renderer, and it works the same
whether the actuation comes from a VR operator, a keyboard, a scripted policy or
a checkpoint under evaluation.

Usage from a sim loop::

    bridge = UnityRenderBridge(env.unwrapped.mujoco, out_dir="data/unity_scene")
    while running:
        observation, ... = env.step(action)
        bridge.tick()
    bridge.close()

``tick()`` is cheap to call every iteration -- it throttles itself to
``publish_hz`` and returns immediately in between.

Scene changes across episodes
-----------------------------
SIMPLE resamples objects per episode, and ``MujocoSimulator._setup_scene``
compiles a fresh ``MjModel`` when it does. The geometry Unity loaded is then
stale, and streaming new poses into it would drive the wrong objects -- body
index 7 might be a mug in one episode and a drill in the next.

The bridge notices by identity: a new ``MjModel`` object means a new scene, so
it re-exports and the ``scene_id`` in the packet header changes with it. A Unity
client that has not reloaded sees the mismatch and can refuse the stream rather
than animate nonsense. Re-export writes to disk and is not instant, so it
happens on the reset boundary rather than mid-episode.
"""

import logging
import time

from .protocol import scene_id_from_names
from .scene_export import body_names, export_scene, world_poses

logger = logging.getLogger(__name__)

DEFAULT_PUBLISH_HZ = 60.0
DEFAULT_EXPORT_DIR = "data/unity_scene"


class UnityRenderBridge:
    """Export a MuJoCo scene to Unity and stream its state.

    Args:
        simulator: a ``MujocoSimulator`` (anything exposing ``mjModel``,
            ``mjData`` and ``render_step``).
        out_dir: where ``scene.json`` and ``meshes/`` are written.
        host, port: signaling endpoint for the Unity client.
        publish_hz: state update rate. 60 matches a typical headset refresh
            closely enough that Unity's own reprojection covers the difference;
            going higher costs bandwidth without being seen. Pass 0 to publish
            on every ``tick`` and let the caller set the pace.
        on_tracker: callback for messages on Unity's ``tracker`` channel.
        server: an already-running state server to publish through, instead of
            starting one. Lets the bridge share a peer connection with a
            signaling server that also carries video.
        include_collision: export collision geometry too, for debugging.
        ice_host: address to advertise for ICE, e.g. this machine's Tailscale
            address. On a multi-homed host aiortc offers a candidate per
            interface and the headset can spend its connection attempt on one
            it cannot reach.
    """

    def __init__(
        self,
        simulator,
        out_dir: str = DEFAULT_EXPORT_DIR,
        host: str = "0.0.0.0",
        port: int = 8765,
        publish_hz: float = DEFAULT_PUBLISH_HZ,
        on_tracker=None,
        server=None,
        include_collision: bool = False,
        ice_host: str | None = None,
    ) -> None:
        self._sim = simulator
        self._out_dir = out_dir
        self._include_collision = include_collision
        self._period = 1.0 / publish_hz if publish_hz > 0 else 0.0
        self._next_publish = 0.0

        self._model = None
        self._scene_id = None
        self.exports = 0

        self._scene_id = self._export()

        self._owns_server = server is None
        if server is None:
            from .webrtc_state import UnityStateServer

            self._server = UnityStateServer(
                scene_id=self._scene_id,
                host=host,
                port=port,
                on_tracker=on_tracker,
                ice_host=ice_host,
            )
        else:
            self._server = server

    # -- scene -------------------------------------------------------------

    def _export(self) -> int:
        model = getattr(self._sim, "mjModel", None)
        if model is None:
            raise RuntimeError(
                "Simulator has no compiled scene yet. mjModel is created by "
                "MujocoSimulator.update_layout(), which the environment calls "
                "from reset() -- build the bridge after the first env.reset()."
            )
        manifest = export_scene(
            model, self._out_dir, include_collision=self._include_collision
        )
        # Hold the reference: it is what makes the identity check below sound.
        # Without it the model could be freed and a new one land at the same
        # address, and the scene change would go unnoticed.
        self._model = model
        self.exports += 1
        logger.info(
            "exported scene to %s (%d bodies, %d geoms, %d meshes, scene_id 0x%08x)",
            self._out_dir,
            len(manifest["bodies"]),
            len(manifest["geoms"]),
            len(manifest["meshes"]),
            manifest["scene_id"],
        )
        return manifest["scene_id"]

    def resync(self) -> bool:
        """Re-export if the simulator has compiled a new scene.

        Called automatically by ``tick``; call it directly right after
        ``env.reset()`` to get the write out of the way before streaming
        resumes.

        Returns:
            True if a re-export happened.
        """
        if self._sim.mjModel is self._model:
            return False

        self._scene_id = self._export()
        self._server.set_scene_id(self._scene_id)
        logger.warning(
            "scene changed; Unity must reload %s (scene_id 0x%08x)",
            self._out_dir,
            self._scene_id,
        )
        return True

    # -- streaming ---------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._server.connected

    @property
    def scene_id(self) -> int:
        return self._scene_id

    def tick(self, force: bool = False) -> bool:
        """Publish the current state if enough time has passed.

        Reads ``xpos``/``xquat`` directly, which ``mj_step`` leaves current --
        no extra ``mj_forward`` is needed on a stepped simulation.

        Returns:
            True if a packet was handed to the transport.
        """
        now = time.monotonic()
        if not force and now < self._next_publish:
            return False
        self._next_publish = now + self._period

        self.resync()

        if not self._server.connected:
            return False

        positions, quaternions = world_poses(self._sim.mjModel, self._sim.mjData)
        return self._server.publish(int(self._sim.render_step), positions, quaternions)

    def stats(self) -> dict:
        stats = dict(self._server.stats())
        stats["exports"] = self.exports
        stats["scene_id"] = self._scene_id
        return stats

    def close(self) -> None:
        if self._owns_server:
            self._server.close()


def scene_id_for(simulator) -> int:
    """Scene id a simulator's current model would export as, without exporting."""
    return scene_id_from_names(body_names(simulator.mjModel))
