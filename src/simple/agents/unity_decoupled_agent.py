"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Teleoperation driven by a Unity XR client, over the same WebRTC connection
that carries the 3D scene.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Why subclass VuerDecoupledAgent
-------------------------------
Most of that agent is WBC assembly with nothing headset-specific in it --
building the observation, the stabilise path, stepping the policy, the elastic
band descent. Only three things actually differ for Unity, so only those are
overridden; the rest is inherited and stays in step if the WBC configuration
changes.

What differs
------------
Input. The poses come from the ``tracker`` data channel rather than a WebXR
browser session, through a ``UnityTrackerSource`` the signaling server feeds.

Buttons. Dropping the band and resetting the episode are simulator concerns
rather than teleoperation ones, and Unity reports its controllers through a
different payload, so the bindings are re-read from the same source.

Rendering. There is none to do. VuerDecoupledAgent resizes a stereo pair and
pushes it to the headset every step; the Unity client receives geometry and
draws the scene itself, which is the entire point of this path. That removes
the per-step ``cv2.resize``, the video buffer, and the coupling between the
operator's viewpoint and the simulator's frame rate.

The device name stays "vuer"
----------------------------
``TeleopPolicy`` is constructed by the inherited method with
``body_control_device="vuer"``, and that is left alone deliberately.
``WristsPreProcessor`` mirrors the right wrist's Z for any device outside its
allowlist, and the name selects a wrist convention rather than a piece of
hardware. Unity's controllers report OpenXR grip poses, the same convention
TeleVuer delivers, so "vuer" selects the correct handling. Renaming it to
"unity" without adding that string to the allowlist would reflect the right arm
alone -- a failure that looks like broken IK rather than a wrong frame.
"""

from __future__ import annotations

from simple.agents.vuer_decoupled_agent import VuerDecoupledAgent
from simple.robots.g1_sonic import G1Sonic
from simple.teleop.unity.streamer import (
    UnityButtonPoller,
    UnityTrackerSource,
    attach_unity_streamer,
)


class UnityDecoupledAgent(VuerDecoupledAgent):
    """Decoupled WBC teleoperation with a Unity client as the XR frontend.

    Args:
        robot: the G1 being controlled.
        source: fed by the signaling server's ``on_tracker`` callback. Passing
            the source rather than the server keeps this agent unaware of the
            transport, so the same agent works whether the channel arrives over
            WebRTC or anything else that can deliver those messages.
    """

    def __init__(self, robot: G1Sonic, source: UnityTrackerSource) -> None:
        self._unity_source = source
        self._buttons = UnityButtonPoller(source)

        # Runs the inherited setup, including the WBC pipeline. The streamer it
        # injects is replaced below rather than prevented: the injection point
        # is the last thing that method does, so overriding the whole method to
        # change one line would duplicate the sixty above it.
        super().__init__(robot)

        self._unity_streamer = attach_unity_streamer(self._teleop_policy, source)
        print("[UnityDecoupled] UnityStreamer injected into TeleopStreamer.")

    # -- input ------------------------------------------------------------

    def _init_vuer_streamer(self) -> None:
        """No WebXR server to start; Unity owns its own XR session.

        ``_tv_wrapper`` is left as None, which the inherited code already
        tolerates: its stereo push is guarded on it, and the two places that
        would dereference it are overridden below.
        """
        self._tv_wrapper = None

    def _poll_vuer_buttons(self) -> None:
        """Read drop and reset from Unity's controller payload."""
        self._buttons.poll()
        # take_drop_request() is leftmost so the pending press is consumed
        # whether or not the band is in a state to act on it; a press that
        # lingered would fire the next time the band happened to be enabled.
        if (
            self._buttons.take_drop_request()
            and self.robot.elastic_band
            and self.robot.elastic_band.enable
            and not self._dropping
        ):
            self._dropping = True
            print("[UnityDecoupled] Controlled drop started.")
        if self._buttons.take_reset_request():
            self._reset_requested = True
            print("[UnityDecoupled] Environment reset requested.")

    # -- rendering --------------------------------------------------------

    def update_render_caches(self, observation: dict) -> dict:
        """Nothing to send. The Unity client renders the scene itself."""
        return observation

    # -- lifecycle --------------------------------------------------------

    def reset_policy(self) -> None:
        super().reset_policy()
        self._buttons.reset_status()

    def close(self) -> None:
        if hasattr(self, "_teleop_policy"):
            self._teleop_policy.close()

    def stats(self) -> dict:
        """Input-side counters, for pairing with the bridge's output ones."""
        return self._unity_source.stats()
