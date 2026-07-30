"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

VuerDecoupledAgent — teleoperation via Meta Quest 3 (or any OpenXR headset).

Drop-in replacement for PicoDecoupledAgent that uses TeleVuer / Vuer
(WebXR over HTTPS) instead of XRoboToolkit / PICO.

Architecture
------------
* TeleVuerWrapper is instantiated here and owns the Vuer server process.
* VuerStreamer (BaseStreamer) wraps the TeleVuerWrapper and is injected into
  TeleopStreamer.body_streamer, bypassing PicoStreamer entirely.
* TeleopPolicy, TeleopStreamer, WristsPreProcessor and TeleopRetargetingIK
  are reused unchanged from decoupled_wbc.
* Video streaming to the headset goes through tv_wrapper.render_to_xr()
  instead of the TCP/H.264 pipeline used by PicoDecoupledAgent.

Button mapping (Meta Quest 3 — see VuerStreamer for full table)
    Drop robot   : Right thumbstick click (right_ctrl_thumbstick, edge)
    Reset env    : Left squeeze + Right squeeze held simultaneously (edge)
    [All teleop/policy/data-collection toggles handled inside VuerStreamer]
"""

import time
import socket
import cv2
import numpy as np

from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic
from simple.teleop.vuer.vuer_streamer import VuerStreamer
from .sonic_wbc_agent import SonicWbcAgent


class VuerDecoupledAgent(SonicWbcAgent):
    """Teleoperation agent using TeleVuer (Meta Quest 3 / OpenXR) + decoupled WBC.

    Mirrors PicoDecoupledAgent but replaces every PICO-specific component:
      - TCPControlServer / TCPVideoSender  →  TeleVuerWrapper
      - PicoStreamer                       →  VuerStreamer (injected)
      - _poll_pico_buttons()              →  _poll_vuer_buttons()

    The WBC pipeline (TeleopPolicy → TeleopRetargetingIK → WBC Policy) is
    inherited unchanged.
    """

    def __init__(self, robot: G1Sonic) -> None:
        super().__init__(robot)

        self.episodes_saved   = 0
        self.num_episodes     = 100
        # Extra HUD lines drawn on the VR stream, refreshed every step by the
        # teleop CLI (e.g. live tote counts) -- empty for tasks that don't set it.
        self.hud_lines: list[str] = []
        self.sim_dt           = self.robot.sonic_config["SIMULATE_DT"]

        # Controlled-drop state
        self._dropping         = False
        self._drop_rate        = 0.15   # m/s
        self._reset_requested  = False
        # Ticks spent under real WBC control (post idle-crouch) with the band
        # still fully up -- used to auto-start the landing sequence only once
        # velocity has genuinely settled here, not just on the first tick
        # `stabilized` happens to latch (see _run_decoupled_policy).
        self._wbc_settle_ticks = 0
        self._auto_drop_settle_ticks = 50   # ~1s at 50Hz control rate
        self._auto_drop_qvel_threshold = 0.1  # max |qvel[0:6]| to allow auto-drop to start

        # Edge-detection for agent-level buttons (drop / reset)
        self._drop_btn_last  = False
        self._reset_btn_last = False

        # 1. Start TeleVuer (Vuer WebXR server)
        self._init_vuer_streamer()

        # 2. Decoupled WBC pipeline (injects VuerStreamer)
        self._init_decoupled_policy()

    # ------------------------------------------------------------------
    # Public property
    # ------------------------------------------------------------------

    @property
    def reset_requested(self) -> bool:
        """True once per reset request (clears on read)."""
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_vuer_port_available(port: int = 8012) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError as exc:
                raise RuntimeError(
                    f"TeleVuer port {port} is already in use. "
                    f"Stop the previous Vuer/TeleVuer process first "
                    f"(e.g. `lsof -i :{port}` then kill the stale PID), then retry."
                ) from exc

    def _init_vuer_streamer(self) -> None:
        """Start the TeleVuerWrapper (Vuer WebXR server process)."""
        from televuer import TeleVuerWrapper

        sonic_cfg = self.robot.sonic_config
        vuer_port = int(sonic_cfg.get("vuer_port", 8012))
        self._assert_vuer_port_available(vuer_port)

        # Image shape is needed even for pass-through mode (internal buffer).
        # Fallback to a safe stereo resolution if not configured.
        img_shape = sonic_cfg.get("vuer_img_shape", (480, 1280))   # (H, W_stereo)
        display_mode = sonic_cfg.get("vuer_display_mode", "immersive")
        cert_file = sonic_cfg.get("vuer_cert_file",  None)
        key_file  = sonic_cfg.get("vuer_key_file",   None)

        self._tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=False,          # controller mode: L/R wrist poses
            binocular=True,
            img_shape=img_shape,
            display_mode=display_mode,
            zmq=True,
            webrtc=False,
            cert_file=cert_file,
            key_file=key_file,
        )
        print(
            f"[VuerDecoupled] TeleVuer started. "
            f"Open https://<PC_IP>:{vuer_port} in the Meta Quest browser."
        )

    def _init_decoupled_policy(self) -> None:
        """Initialise the decoupled WBC pipeline, injecting VuerStreamer."""
        from decoupled_wbc.control.robot_model.instantiation.g1 import (
            instantiate_g1_robot_model,
        )
        from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
            instantiate_g1_hand_ik_solver,
        )
        from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
        from decoupled_wbc.control.policy.teleop_policy import TeleopPolicy
        from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
        from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig

        sonic_cfg = self.robot.sonic_config
        enable_waist      = sonic_cfg.get("enable_waist", False)
        waist_location    = "lower_and_upper_body" if enable_waist else "lower_body"

        self._dwbc_robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=sonic_cfg.get("high_elbow_pose", False),
        )

        supp = self._dwbc_robot_model.supplemental_info
        assert supp.body_actuated_joints   == self.robot.joint_names[:29]
        assert supp.left_hand_actuated_joints  == self.robot.hand_names[:7]
        assert supp.right_hand_actuated_joints == self.robot.hand_names[7:14]

        dwbc_config = ControlLoopConfig(
            enable_waist=enable_waist,
            high_elbow_pose=sonic_cfg.get("high_elbow_pose", False),
        )
        wbc_config = dwbc_config.load_wbc_yaml()
        assert wbc_config["SIMULATE_DT"] == self.sim_dt

        self._wbc_policy = get_wbc_policy(
            "g1", self._dwbc_robot_model, wbc_config,
            init_time=dwbc_config.upper_body_joint_speed,
        )

        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        retargeting_ik = TeleopRetargetingIK(
            robot_model=self._dwbc_robot_model,
            left_hand_ik_solver=left_hand_ik,
            right_hand_ik_solver=right_hand_ik,
            body_active_joint_groups=["upper_body"],
        )

        # Instantiate TeleopPolicy with device="vuer" — an unknown string that
        # falls into TeleopStreamer's else-branch (body_streamer = None),
        # avoiding DummyStreamer which requires ROS2 (KeyboardListenerSubscriber).
        # enable_real_device defaults to True so _get_live_data() is called,
        # which will use our injected VuerStreamer via body_streamer.get().
        self._teleop_policy = TeleopPolicy(
            robot_model=self._dwbc_robot_model,
            retargeting_ik=retargeting_ik,
            body_control_device="vuer",
            hand_control_device="vuer",
            body_streamer_ip="",
            activate_keyboard_listener=False,
        )

        # Inject VuerStreamer — TeleopStreamer already has body_streamer=None at
        # this point (no DummyStreamer, no ROS2). Injection makes _get_live_data()
        # call vuer_streamer.get() on every control step.
        vuer_streamer = VuerStreamer(self._tv_wrapper)
        vuer_streamer.start_streaming()
        self._teleop_policy.teleop_streamer.body_streamer = vuer_streamer
        self._teleop_policy.teleop_streamer.hand_streamer = vuer_streamer
        print("[VuerDecoupled] VuerStreamer injected into TeleopStreamer.")

        self._teleop_initialized   = False
        self._teleop_was_active    = False
        self._teleop_activate_time: float | None = None
        self._t_start              = time.monotonic()
        self._control_frequency    = dwbc_config.control_frequency
        self._arm_engage_smooth_secs = 1.0
        self._control_dt           = 4 * self.sim_dt   # 50 Hz
        self._cached_target_q      = None
        self._cached_left_hand_q   = None
        self._cached_right_hand_q  = None
        self._last_teleop_action   = {}

    # ------------------------------------------------------------------
    # Button polling (agent-level: drop / reset only)
    # ------------------------------------------------------------------

    def _poll_vuer_buttons(self) -> None:
        """Detect edge-triggered drop/reset commands from Meta Quest buttons.

        These are SIMPLE-level concerns (elastic band, env reset) and are
        intentionally not delegated to VuerStreamer.

        Mapping:
            drop robot  : Right thumbstick click (right_ctrl_thumbstick, edge)
            reset env   : Left squeeze + Right squeeze held simultaneously (edge)
        """
        td = self._tv_wrapper.get_tele_data()

        # --- drop_robot: right thumbstick click (edge) ---
        drop_btn = bool(td.right_ctrl_thumbstick)
        if drop_btn and not self._drop_btn_last:
            if (
                self.robot.elastic_band
                and self.robot.elastic_band.enable
                and not self._dropping
            ):
                self._dropping = True
                print("[VuerDecoupled] Controlled drop started (right thumbstick click).")
        self._drop_btn_last = drop_btn

        # --- reset_env: both squeezes held (edge) ---
        left_sq  = float(td.left_ctrl_squeezeValue)  > 0.5
        right_sq = float(td.right_ctrl_squeezeValue) > 0.5
        reset_btn = left_sq and right_sq
        if reset_btn and not self._reset_btn_last:
            self._reset_requested = True
            print("[VuerDecoupled] Environment reset requested (L-squeeze + R-squeeze).")
        self._reset_btn_last = reset_btn

    # ------------------------------------------------------------------
    # Rendering / streaming
    # ------------------------------------------------------------------

    def update_render_caches(self, observation: dict) -> dict:
        """Push stereo frame to the VR headset via TeleVuer."""
        if self._tv_wrapper is not None:
            self._push_stereo_frame(observation)
        return observation

    def _push_stereo_frame(self, observation: dict) -> None:
        left  = observation.get("head_stereo_left")
        right = observation.get("head_stereo_right")
        if left is None or right is None:
            print("[VuerDecoupled] Warning: missing head_stereo_left/right in observation.")
            return

        left_bgr  = np.ascontiguousarray(left[..., ::-1])
        right_bgr = np.ascontiguousarray(right[..., ::-1])

        # Draw episode counter on each eye
        text   = f"{self.episodes_saved}/{self.num_episodes}"
        font   = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 1.0, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = left_bgr.shape[1] - tw - 100
        y = th + 10
        cv2.putText(left_bgr, text, (x, y), font, scale, (0, 255, 0), thickness)
        cv2.putText(right_bgr, text, (x, y), font, scale, (0, 255, 0), thickness)

        # Extra per-step HUD lines (e.g. live tote counts), top-left
        hud_scale, hud_thickness = 0.7, 2
        for i, line in enumerate(self.hud_lines):
            y_pos = 30 + i * 28
            cv2.putText(left_bgr, line, (20, y_pos), font, hud_scale, (0, 255, 255), hud_thickness)
            cv2.putText(right_bgr, line, (20, y_pos), font, hud_scale, (0, 255, 255), hud_thickness)

        stereo_bgr = np.concatenate([left_bgr, right_bgr], axis=1)
        
        # Resize to match TeleVuer's internal shared memory buffer expected shape (H, W)
        target_h, target_w, _ = self._tv_wrapper.tvuer.img_shape
        if stereo_bgr.shape[:2] != (target_h, target_w):
            stereo_bgr = cv2.resize(stereo_bgr, (target_w, target_h))

        # TeleVuer converts BGR→RGB internally before sending via WebXR
        self._tv_wrapper.render_to_xr(stereo_bgr)

    # ------------------------------------------------------------------
    # WBC observation builder (identical to PicoDecoupledAgent)
    # ------------------------------------------------------------------

    def _build_wbc_observation(self, sim_obs: dict) -> dict:
        rm  = self._dwbc_robot_model
        obs = {}

        left_hand_q   = sim_obs.get("left_hand_q",   np.zeros(7))
        right_hand_q  = sim_obs.get("right_hand_q",  np.zeros(7))
        left_hand_dq  = sim_obs.get("left_hand_dq",  np.zeros(7))
        right_hand_dq = sim_obs.get("right_hand_dq", np.zeros(7))

        mjcf_to_natural = lambda q: np.concatenate([q[:3], q[5:7], q[3:5]])

        obs["q"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs["body_q"],
            left_hand_actuated_joint_values=mjcf_to_natural(left_hand_q),
            right_hand_actuated_joint_values=mjcf_to_natural(right_hand_q),
        )
        obs["dq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs["body_dq"],
            left_hand_actuated_joint_values=mjcf_to_natural(left_hand_dq),
            right_hand_actuated_joint_values=mjcf_to_natural(right_hand_dq),
        )
        obs["ddq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_ddq", np.zeros(29)),
            left_hand_actuated_joint_values=mjcf_to_natural(sim_obs.get("left_hand_ddq", np.zeros(7))),
            right_hand_actuated_joint_values=mjcf_to_natural(sim_obs.get("right_hand_ddq", np.zeros(7))),
        )
        obs["tau_est"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_tau_est", np.zeros(29)),
            left_hand_actuated_joint_values=mjcf_to_natural(sim_obs.get("left_hand_tau_est", np.zeros(7))),
            right_hand_actuated_joint_values=mjcf_to_natural(sim_obs.get("right_hand_tau_est", np.zeros(7))),
        )

        obs["floating_base_pose"] = sim_obs["floating_base_pose"]
        obs["floating_base_vel"]  = sim_obs["floating_base_vel"]
        obs["floating_base_acc"]  = sim_obs.get("floating_base_acc", np.zeros(6))

        obs["torso_quat"]    = sim_obs.get("secondary_imu_quat", np.array([1, 0, 0, 0]))
        obs["torso_ang_vel"] = (
            sim_obs.get("secondary_imu_vel", np.zeros(6))[3:6]
            if "secondary_imu_vel" in sim_obs
            else np.zeros(3)
        )
        obs["wrist_pose"] = sim_obs.get("wrist_pose", np.zeros(14))
        return obs

    # ------------------------------------------------------------------
    # Stabilise action (arm → default pose before teleop activation)
    # ------------------------------------------------------------------

    def get_stabilize_action(self, proprio) -> ActionCmd:
        from decoupled_wbc.control.main.constants import DEFAULT_BASE_HEIGHT, DEFAULT_NAV_CMD

        t_now = time.monotonic()
        wbc_obs = self._build_wbc_observation(self.robot.prepare_obs())
        self._wbc_policy.set_observation(wbc_obs)

        is_first_step = self._cached_target_q is None
        default_upper = self._dwbc_robot_model.get_initial_upper_body_pose()
        goal = {
            "target_upper_body_pose":              default_upper,
            "navigate_cmd":                        np.asarray(DEFAULT_NAV_CMD),
            "base_height_command":                 np.atleast_1d(np.asarray(DEFAULT_BASE_HEIGHT)),
            "target_time":                         t_now + (2.0 if is_first_step else 1 / self._control_frequency),
            "interpolation_garbage_collection_time": t_now - 2 / self._control_frequency,
            "timestamp":                           t_now,
        }
        self._wbc_policy.set_goal(goal)

        wbc_action = self._wbc_policy.get_action(time=t_now)
        self._cached_target_q      = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        self._cached_left_hand_q   = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="left")
        self._cached_right_hand_q  = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="right")

        return ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            left_hand_q=self._cached_left_hand_q,
            right_hand_q=self._cached_right_hand_q,
        )

    # ------------------------------------------------------------------
    # Core: run decoupled WBC pipeline
    # ------------------------------------------------------------------

    def _run_decoupled_policy(self, sim_obs: dict) -> dict:
        t_now = time.monotonic()

        teleop_action = self._teleop_policy.get_action()
        self._last_teleop_action = teleop_action
        synced_eef = teleop_action["wrist_pose"]

        teleop_action["timestamp"] = t_now
        first_wbc_call = not self._teleop_initialized
        if first_wbc_call:
            teleop_action["target_time"] = t_now + 2.0
            self._teleop_initialized = True
            self._wbc_settle_ticks = 0
        else:
            teleop_action["target_time"] = t_now + 1 / self._control_frequency

        # Auto-start the landing sequence once real WBC control has held a
        # genuinely settled stance for a short buffer, instead of waiting for
        # a manual drop-button press. Previously the robot would linger
        # suspended near the band's anchor (z=1.0) indefinitely -- fighting
        # the WBC policy's own attempt to plant its feet ("kicking, searching
        # for the ground") -- until the operator pressed the right-thumbstick
        # drop button. Triggering this on the very *first* WBC tick (tried
        # first) was too eager: `robot.stabilized` only requires one momentary
        # dip below threshold, so the drop could start while qvel was still
        # settling, producing a fast/violent release. Requiring both a short
        # settle window *and* a fresh low-velocity check right before
        # triggering is much closer to what a human operator does by eye
        # before pressing the button.
        if not self._dropping and self.robot.elastic_band and self.robot.elastic_band.enable:
            self._wbc_settle_ticks += 1
            qvel_max = float(np.max(np.abs(self.robot.mjData.qvel[0:6])))
            if self._wbc_settle_ticks >= self._auto_drop_settle_ticks and qvel_max < self._auto_drop_qvel_threshold:
                self._dropping = True
                print(
                    f"[VuerDecoupled] Landing sequence started automatically "
                    f"(settled {self._wbc_settle_ticks} ticks, qvel_max={qvel_max:.3f})."
                )

        wbc_obs = self._build_wbc_observation(sim_obs)
        self._wbc_policy.set_observation(wbc_obs)

        teleop_just_activated = self._teleop_policy.is_active and not self._teleop_was_active
        self._teleop_was_active = self._teleop_policy.is_active

        # Treat the very first decoupled-WBC call the same as an engagement
        # edge: get_stabilize_action() was driving a completely different
        # default-pose goal up until now, so the teleop policy's own
        # idle/home target can be far from the robot's actual current joint
        # configuration. Without this, that first call seeds `q` straight
        # from the teleop policy's idle target with no reference to where
        # the robot physically is, producing a violent, self-colliding pose
        # jump (knee-to-torso, shoulder-to-ankle contacts observed) before
        # the sim can catch up -- see the migration plan doc, "Real-stack
        # validation", for how this was diagnosed.
        if first_wbc_call or teleop_just_activated:
            self._teleop_activate_time = t_now
            if teleop_just_activated:
                print("[VuerDecoupled] Teleop activated — smoothing arm engagement.")
            else:
                print("[VuerDecoupled] First WBC control step — smoothing pose handoff from idle stance.")

        wbc_goal = {}
        if teleop_action:
            wbc_goal = teleop_action.copy()
            wbc_goal["interpolation_garbage_collection_time"] = (
                t_now - 2 * (1 / self._control_frequency)
            )
            if self._teleop_activate_time is not None:
                elapsed   = t_now - self._teleop_activate_time
                remaining = max(0.0, self._arm_engage_smooth_secs - elapsed)
                if remaining > 0.0:
                    wbc_goal["target_time"] = t_now + remaining
            if (first_wbc_call or teleop_just_activated) and "q" in wbc_goal:
                wbc_goal["q"] = wbc_obs["q"].copy()

        if wbc_goal:
            self._wbc_policy.set_goal(wbc_goal)

        wbc_action = self._wbc_policy.get_action(time=t_now)
        wbc_action.update({"action_eef": synced_eef})
        return wbc_action

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def get_action(self, observation, instruction=None, **kwargs) -> ActionCmd:
        proprio = kwargs["privileged_info"]["proprio"]

        if not self.robot.stabilized:
            return self.get_stabilize_action(proprio)

        wbc_action = self._run_decoupled_policy(proprio)
        self._poll_vuer_buttons()

        self._cached_target_q     = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        self._cached_left_hand_q  = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="left")
        self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="right")

        # Elastic band descent
        if self._dropping and self.robot.elastic_band and self.robot.elastic_band.enable:
            self.robot.elastic_band.length -= self._drop_rate * self._control_dt
            if self.robot.elastic_band.length <= -0.25 and abs(self.robot.pelvis_vz) < 0.05:
                self.robot.elastic_band.enable = False
                self._dropping = False
                print(f"[VuerDecoupled] Robot landed (pelvis Z={self.robot.pelvis_z:.3f} m).")

        if (
            self.robot.elastic_band
            and self.robot.elastic_band.enable
            and self.robot.use_floating_root_link
        ):
            return ActionCmd(
                "elastic_band",
                dropping=self._dropping,
                drop_rate=self._drop_rate,
                # Keep the joints actively PD-tracking the current WBC target
                # while the band holds the pelvis -- without this, g1_sonic's
                # apply_action() previously left mjData.ctrl frozen at
                # whatever it was the instant this action type took over from
                # "decoupled_wbc", fighting the band's strong external force
                # with stale torques (see the migration plan doc, "Real-stack
                # validation", for how this was diagnosed).
                target_q=self._cached_target_q,
                left_hand_q=self._cached_left_hand_q,
                right_hand_q=self._cached_right_hand_q,
            )

        return ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            left_hand_q=self._cached_left_hand_q,
            right_hand_q=self._cached_right_hand_q,
            base_height_command=wbc_action["base_height_command"],
            navigate_cmd=wbc_action["navigate_cmd"],
            torso_rpy_cmd=wbc_action["torso_rpy_cmd"],
            action_eef=wbc_action["action_eef"],
            obs_tensor=wbc_action["obs_tensor"],
        )

    def reset_policy(self) -> None:
        """Reset the WBC pipeline for a new episode."""
        t_now = time.monotonic()
        self._wbc_policy.reset(init_time=t_now)
        self._teleop_policy.reset()

        self._cached_target_q      = None
        self._cached_left_hand_q   = None
        self._cached_right_hand_q  = None
        self._teleop_initialized   = False
        self._teleop_was_active    = False
        self._teleop_activate_time = None
        self._last_teleop_action   = {}
        self._wbc_settle_ticks     = 0
        self._dropping             = False

        # Reset VuerStreamer internal state (height, yaw, edge detectors)
        self._teleop_policy.teleop_streamer.body_streamer.reset_status()

    def publish_low_state(self, proprio) -> None:
        # No Unitree bridge needed for simulation
        pass

    def close(self) -> None:
        if hasattr(self, "_teleop_policy"):
            self._teleop_policy.close()
        if hasattr(self, "_tv_wrapper"):
            self._tv_wrapper.close()
