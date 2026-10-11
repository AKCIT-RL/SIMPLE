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

Why the wrist alignment is computed, not named
----------------------------------------------
``TeleopStreamer`` recognises neither, which is what both agents want: an
unrecognised name leaves ``body_streamer`` as None, so nothing is built for us
to fight with and no ``DummyStreamer`` drags in ROS 2. That much they share.

``WristsPreProcessor.calibrate`` does read the name, and this is the part that
matters. It is a differential controller: at calibration it stores the
operator's wrist pose, and every frame after it applies ``inv(wrist_at_calib) @
wrist_now`` to the robot's hand *in that hand's own frame*. So a hand
translation reaches the robot as ``R_hand_frame @ R_wrist.T @ delta`` -- it is
routed through the operator's wrist orientation, and only comes out pointing
the right way if that orientation matches the robot's hand frame.

The names "pico" and "vuer" assert exactly that match, and the preprocessor
applies no correction for them. Any other name makes it apply a hardcoded pair
instead -- but that pair is an assertion about a different headset, so it is
another guess rather than an answer. Unity reports ``XRNode`` device poses,
matching neither.

The asymmetry is the tell. The G1's two hand frames are mirror images of each
other, so one wrong wrist convention -- identical on both hands -- lands
differently on each arm, and no single constant can be right for both.

So this agent computes the alignment rather than naming it. Both matrices it
needs are already stored by ``calibrate()``, and ``wrist_alignment`` derives
the one rotation that makes each arm follow the operator's hand 1:1; see that
module for the derivation. The device name stays "vuer", leaving the
preprocessor's own choice at identity so nothing has to be undone.

Pass ``wrist_correction=False`` to skip the alignment and compare.
"""

from __future__ import annotations

from simple.agents.vuer_decoupled_agent import VuerDecoupledAgent
from simple.robots.g1_sonic import G1Sonic
from simple.teleop.unity.streamer import (
    UnityButtonPoller,
    UnityTrackerSource,
    attach_unity_streamer,
)
from simple.teleop.unity.wrist_alignment import align_wrist_frames


class UnityDecoupledAgent(VuerDecoupledAgent):
    """Decoupled WBC teleoperation with a Unity client as the XR frontend.

    Args:
        robot: the G1 being controlled.
        source: fed by the signaling server's ``on_tracker`` callback. Passing
            the source rather than the server keeps this agent unaware of the
            transport, so the same agent works whether the channel arrives over
            WebRTC or anything else that can deliver those messages.
    """

    def __init__(
        self,
        robot: G1Sonic,
        source: UnityTrackerSource,
        wrist_correction: bool = True,
    ) -> None:
        # Exposed because confirming it needs a headset and a robot, and an
        # edit-and-rebuild per comparison is a poor trade for one argument.
        self._wrist_correction = wrist_correction

        self._unity_source = source
        self._buttons = UnityButtonPoller(source)

        # Runs the inherited setup, including the WBC pipeline. The streamer it
        # injects is replaced below rather than prevented: the injection point
        # is the last thing that method does, so overriding the whole method to
        # change one line would duplicate the sixty above it.
        super().__init__(robot)

        self._unity_streamer = attach_unity_streamer(self._teleop_policy, source)
        print("[UnityDecoupled] UnityStreamer injected into TeleopStreamer.")

        if wrist_correction:
            self._install_wrist_alignment()

    # -- wrist frames -----------------------------------------------------

    def _install_wrist_alignment(self) -> None:
        """Solve the alignment each time the operator calibrates.

        Wrapping ``calibrate`` rather than calling ``align_wrist_frames`` once
        here, because the alignment depends on where the operator's hands were
        when they pressed the button -- so it has to be recomputed on every
        activation, not fixed at startup.
        """
        streamer = self._teleop_policy.teleop_streamer
        calibrate = streamer.calibrate

        def calibrate_and_align():
            calibrate()
            pre_processor = streamer.body_pre_processor
            if pre_processor is None:
                return
            aligned = align_wrist_frames(pre_processor)
            for ee_name in aligned:
                print(f"[UnityDecoupled] Wrist frame aligned for {ee_name}.")

        streamer.calibrate = calibrate_and_align

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
