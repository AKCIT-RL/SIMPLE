"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

VuerStreamer — BaseStreamer implementation backed by TeleVuerWrapper.

Produces a StreamerOutput with the exact same contract as PicoStreamer so that
TeleopStreamer / WristsPreProcessor / TeleopRetargetingIK can be reused intact.

Key design decisions
--------------------
* Wrist poses: obtained via TeleVuerWrapper.get_headset_relative_wrist_poses(),
  which converts raw OpenXR 4×4 matrices to z-up, headset-relative,
  yaw-compensated SE3 — the format expected by WristsPreProcessor.
* Finger data: generated from trigger/squeeze booleans (Meta Quest controllers),
  replicating the same (25,4,4) convention as PicoStreamer._generate_finger_data().
* Navigation: thumbstick values → lin_vel_x/y + integrated target_yaw,
  with the same dead-zone logic and 50 Hz assumption as PicoStreamer.
* All toggle signals (activation, policy, data collection) use edge-detection
  (False→True transition) to match PicoStreamer behaviour.

Button mapping (Meta Quest 3 controllers → PICO equivalents)
-------------------------------------------------------------
  Toggle teleop activation  : Left  B button (edge)       ← left_menu+right_trigger
  Toggle policy action      : Left  A button (edge)       ← left_menu+left_trigger
  Height increase           : Right A button (hold)       ← Y button
  Height decrease           : Right B button (hold)       ← X button
  Data collection toggle    : Right trigger   (edge)      ← A button
  Data abort                : Right squeeze   (edge)      ← B button
  [Drop robot / Reset env handled at agent level, not here]
"""

import time

import numpy as np

from decoupled_wbc.control.teleop.streamers.base_streamer import BaseStreamer, StreamerOutput


class VuerStreamer(BaseStreamer):
    """Streamer backed by TeleVuerWrapper — supports Meta Quest 3 and any OpenXR headset.

    Args:
        tv_wrapper: A fully initialised ``TeleVuerWrapper`` instance whose
                    underlying ``TeleVuer`` process is already running.
    """

    # Navigation constants (match PicoStreamer)
    _DEAD_ZONE        = 0.1
    _MAX_LINEAR_VEL   = 0.8   # m/s
    _MAX_ANGULAR_VEL  = 1.0   # rad/s
    _CONTROL_DT       = 1.0 / 50.0  # fixed 50 Hz call-rate assumption

    # Base height limits (match PicoStreamer)
    _HEIGHT_INCREMENT = 0.01
    _HEIGHT_MIN       = 0.20
    _HEIGHT_MAX       = 0.74
    _HEIGHT_DEFAULT   = 0.74

    def __init__(self, tv_wrapper) -> None:
        super().__init__()
        self._tv = tv_wrapper
        self.reset_status()

    # ------------------------------------------------------------------
    # BaseStreamer interface
    # ------------------------------------------------------------------

    def start_streaming(self) -> None:
        # TeleVuer server is started by TeleVuerWrapper at construction time.
        pass

    def stop_streaming(self) -> None:
        # Lifecycle managed by VuerDecoupledAgent (calls tv_wrapper.close()).
        pass

    def reset_status(self, initial_yaw: float = 0.0) -> None:
        """Reset internal state — called on episode reset.

        `target_yaw` is a world-frame heading the yaw PD controller in
        `G1GearWbcPolicy` tracks (see `g1_gear_wbc_policy.py`); it must start
        at the robot's actual spawn yaw, not always 0.0 -- otherwise, on any
        task whose robot spawn orientation isn't identity, the controller
        reads a spurious yaw error at episode start and spins the robot back
        toward world yaw 0 ("the old orientation").
        """
        self.current_base_height = self._HEIGHT_DEFAULT
        self.target_yaw = initial_yaw
        self._last_valid_left_wrist = np.eye(4, dtype=np.float64)
        self._last_valid_right_wrist = np.eye(4, dtype=np.float64)
        self._last_pose_warning_time = 0.0

        # Edge-detection state for all toggle buttons
        self._toggle_activation_last        = False
        self._toggle_policy_action_last     = False
        self._toggle_data_collection_last   = False
        self._toggle_data_abort_last        = False

    def get(self) -> StreamerOutput:
        """Read one frame from TeleVuer and return a StreamerOutput."""
        # ------------------------------------------------------------------
        # 1. Wrist poses — headset-relative z-up (for WristsPreProcessor)
        # ------------------------------------------------------------------
        left_wrist, right_wrist = self._safe_get_wrist_poses()

        # ------------------------------------------------------------------
        # 2. Controller state — read once via get_tele_data() for buttons
        # ------------------------------------------------------------------
        td = self._tv.get_tele_data()

        # ------------------------------------------------------------------
        # 3. Finger data  (25,4,4) — same convention as PicoStreamer
        # ------------------------------------------------------------------
        left_fingers  = self._generate_finger_data(td, "left")
        right_fingers = self._generate_finger_data(td, "right")

        # ------------------------------------------------------------------
        # 4. Navigation (thumbstick → lin/ang velocity + integrated yaw)
        # ------------------------------------------------------------------
        fwd_bwd = -td.left_ctrl_thumbstickValue[1]   # −y → forward positive
        strafe  = -td.left_ctrl_thumbstickValue[0]   # −x → left positive
        yaw_in  = -td.right_ctrl_thumbstickValue[0]  # −x → CCW positive

        lin_vel_x = self._apply_dead_zone(fwd_bwd, self._DEAD_ZONE) * self._MAX_LINEAR_VEL
        lin_vel_y = self._apply_dead_zone(strafe,  self._DEAD_ZONE) * self._MAX_LINEAR_VEL
        vyaw      = self._apply_dead_zone(yaw_in,  self._DEAD_ZONE) * self._MAX_ANGULAR_VEL

        self.target_yaw += vyaw * self._CONTROL_DT
        # Wrap to [-π, π]
        self.target_yaw = np.arctan2(np.sin(self.target_yaw), np.cos(self.target_yaw))

        # ------------------------------------------------------------------
        # 5. Base height (hold right A/B)
        # ------------------------------------------------------------------
        if td.right_ctrl_aButton:
            self.current_base_height += self._HEIGHT_INCREMENT
        elif td.right_ctrl_bButton:
            self.current_base_height -= self._HEIGHT_INCREMENT
        self.current_base_height = float(
            np.clip(self.current_base_height, self._HEIGHT_MIN, self._HEIGHT_MAX)
        )

        # ------------------------------------------------------------------
        # 6. Edge-detected toggle signals
        # ------------------------------------------------------------------
        # toggle_activation  — Left B button
        toggle_activation_now = bool(td.left_ctrl_bButton)
        toggle_activation     = toggle_activation_now and not self._toggle_activation_last
        self._toggle_activation_last = toggle_activation_now

        # toggle_policy_action — Left A button
        toggle_policy_now  = bool(td.left_ctrl_aButton)
        toggle_policy      = toggle_policy_now and not self._toggle_policy_action_last
        self._toggle_policy_action_last = toggle_policy_now

        # toggle_data_collection — Right trigger (bool)
        toggle_dc_now = bool(td.right_ctrl_trigger)
        toggle_dc     = toggle_dc_now and not self._toggle_data_collection_last
        self._toggle_data_collection_last = toggle_dc_now

        # toggle_data_abort — Right squeeze (bool)
        toggle_da_now = bool(td.right_ctrl_squeeze)
        toggle_da     = toggle_da_now and not self._toggle_data_abort_last
        self._toggle_data_abort_last = toggle_da_now

        # ------------------------------------------------------------------
        # 7. Assemble StreamerOutput
        # ------------------------------------------------------------------
        return StreamerOutput(
            ik_data={
                "left_wrist":   left_wrist,
                "right_wrist":  right_wrist,
                "left_fingers": {"position": left_fingers},
                "right_fingers":{"position": right_fingers},
            },
            control_data={
                "base_height_command": self.current_base_height,
                "navigate_cmd":        [lin_vel_x, lin_vel_y, vyaw, self.target_yaw],
                "toggle_policy_action": toggle_policy,
            },
            teleop_data={
                "toggle_activation": toggle_activation,
            },
            data_collection_data={
                "toggle_data_collection": toggle_dc,
                "toggle_data_abort":      toggle_da,
            },
            source="vuer",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_finger_data(self, tele_data, side: str) -> np.ndarray:
        """Build a (25,4,4) finger SE3 array from controller trigger/squeeze.

        Replicates PicoStreamer._generate_finger_data() using Meta Quest inputs:
          - trigger bool  ↔  pico_data["<side>_trigger"] > 0.5
          - squeeze float ↔  pico_data["<side>_grip"]    > 0.5

        Finger-tip indices follow the PicoStreamer convention:
          thumb=0, index=5, middle=10, ring=15  (+4 for the tip within each group).
        The flag is stored in column [0], translation component [3]:
          1.0 → closed, 0.0 → open.
        """
        fingertips = np.zeros((25, 4, 4))
        THUMB = 0; INDEX = 5; MIDDLE = 10; RING = 15

        trigger = bool(getattr(tele_data, f"{side}_ctrl_trigger"))
        squeeze = float(getattr(tele_data, f"{side}_ctrl_squeezeValue")) > 0.5

        # Thumb always open (matches PicoStreamer default)
        fingertips[4 + THUMB, 0, 3] = 1.0

        if trigger and not squeeze:
            fingertips[4 + INDEX, 0, 3]  = 1.0   # close index
        elif trigger and squeeze:
            fingertips[4 + MIDDLE, 0, 3] = 1.0   # close middle
        elif not trigger and squeeze:
            fingertips[4 + RING, 0, 3]   = 1.0   # close ring

        return fingertips

    @staticmethod
    def _apply_dead_zone(value: float, dead_zone: float) -> float:
        """Apply dead-zone and renormalize to [-1, 1].  Matches PicoStreamer."""
        if abs(value) < dead_zone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - dead_zone) / (1.0 - dead_zone)

    def _safe_get_wrist_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Return valid SE3 wrist poses or fall back to last valid values."""
        try:
            left_wrist, right_wrist = self._tv.get_headset_relative_wrist_poses()
        except Exception as exc:
            self._maybe_warn_invalid_pose(f"exception reading wrist poses: {exc}")
            return self._last_valid_left_wrist, self._last_valid_right_wrist

        if not self._is_valid_se3(left_wrist) or not self._is_valid_se3(right_wrist):
            self._maybe_warn_invalid_pose("received invalid wrist pose matrix, using fallback")
            return self._last_valid_left_wrist, self._last_valid_right_wrist

        self._last_valid_left_wrist = left_wrist
        self._last_valid_right_wrist = right_wrist
        return left_wrist, right_wrist

    @staticmethod
    def _is_valid_se3(mat: np.ndarray) -> bool:
        if not isinstance(mat, np.ndarray) or mat.shape != (4, 4):
            return False
        if not np.all(np.isfinite(mat)):
            return False
        if not np.allclose(mat[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
            return False
        det = float(np.linalg.det(mat[:3, :3]))
        return np.isfinite(det) and det > 0.0 and np.isclose(det, 1.0, atol=1e-2)

    def _maybe_warn_invalid_pose(self, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_pose_warning_time > 2.0:
            print(f"[VuerStreamer] Warning: {reason}")
            self._last_pose_warning_time = now
