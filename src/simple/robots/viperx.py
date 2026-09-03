"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from typing import Any, List, Tuple

import numpy as np
import transforms3d as t3d

from simple.core.controller import ControllerCfg
from simple.core.robot import Robot
from simple.robots.controllers.combo import SingleArmBinaryEEFController, SingleArmBinaryEEFControllerCfg
from simple.robots.controllers.eef import ParallelGripperEEFControllerCfg
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.mixin import CuRoboMixin
from simple.robots.protocols import HasParallelGripper, WristCamMountable
from simple.robots.registry import RobotRegistry


@RobotRegistry.register("viperx")
class ViperX(CuRoboMixin, WristCamMountable, HasParallelGripper, Robot):
    """Single-arm Trossen ViperX robot, extracted from Aloha's left arm.

    Aloha (`src/simple/robots/aloha.py`) is two independently-rooted ViperX arms sharing one
    `Robot` class (`DualArm` protocol, `active_arm`/`switch_arm()`, a `DualArmBinaryEEFController`
    that parks the inactive arm's ctrl every step). That coupling is Aloha-specific, not
    ViperX-hardware-specific -- the left arm's MJCF/URDF body tree, curobo collision spheres, and
    grasp-frame convention are already independent of the right arm (see
    docs/source/viperx/viperx_integration.md). This class re-exposes just the left arm as its own
    single-arm embodiment, modeled directly on `WidowXAI` (`src/simple/robots/widowx_ai.py`)
    rather than on `Aloha`.
    """

    uid: str = "viperx"
    label: str = "Trossen ViperX (single-arm, from Aloha)"
    # 6 arm joints + 2 independently-actuated finger joints (Aloha's gripper has no mimic joint,
    # unlike WidowX AI's carriage joints -- both left_left_finger/left_right_finger are real DOFs).
    dof: int = 8

    robot_cfg: Any = "robots/viperx/curobo/viperx.yml"
    mjcf_path: str = "robots/viperx/viperx.xml"
    # No single-arm USD exists (Aloha's own aloha.usd is a fused dual-arm asset) -- placeholder
    # for API symmetry only, matching WidowXAI's precedent. Only sim_mode="mujoco" is supported.
    usd_path: str = "robots/viperx/viperx.usd"

    robot_ns: str = "viperx"
    hand_prim_path: str = "left_gripper_base"
    eef_prim_path: str = "left_gripper_site"
    wrist_cam_link: str = "left_gripper_camera"

    # Aloha's own wrist camera orientation constant for its per-arm wrist cam mount.
    wrist_camera_orientation: List[float] = [0.5, 0.5, -0.5, 0.5]

    pregrasp_distance: List[float] = [0.05, 0.08]
    visulize_spheres: bool = False
    # Reused as-is from Aloha.robot_eef_offset -- Aloha's own comment marks this "HACK to avoid
    # collision" rather than a value derived from link geometry; not re-derived here.
    robot_eef_offset: float = 0.00

    joint_names: List[str] = [
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
        "left_wrist_angle", "left_wrist_rotate", "left_left_finger", "left_right_finger",
    ]
    # User-specified rest pose for the tabletop grasp task (overrides the earlier Aloha-derived
    # rest pose). Gripper fingers are set to their joint range's own max (0.041, see viperx.xml's
    # `left_finger`/`right_finger` default classes, range="0 0.041") so the gripper starts fully
    # open, per request.
    init_joint_states: dict[str, float] = {
        "left_waist": 3.1377,
        "left_shoulder": -1.941,
        "left_elbow": 0.5015,
        "left_forearm_roll": -0.0,
        "left_wrist_angle": 0.907,
        "left_wrist_rotate": -0.0054,
        "left_left_finger": 0.041,
        "left_right_finger": 0.041,
    }

    controller_cfg: ControllerCfg = SingleArmBinaryEEFControllerCfg(
        arm=PDJointPosControllerCfg(
            joint_names=["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate"],
            init_qpos=[3.1377, -1.941, 0.5015, -0.0, 0.907, -0.0054],
        ),
        # Generic ParallelGripperEEFController (not a WidowX-style subclass): Aloha's own
        # left_eef already uses this same generic controller with the same joint names --
        # ViperX is the identical gripper hardware, so its hardcoded open/close ctrl values
        # (0.0205 / 0.0, see src/simple/robots/controllers/eef.py) apply unchanged. init_qpos is
        # set to the joint range's own max (0.041) so the gripper starts fully open, per request.
        eef=ParallelGripperEEFControllerCfg(
            joint_names=["left_left_finger", "left_right_finger"],
            init_qpos=[0.041, 0.041],
        ),
    )

    def __init__(self):
        super().__init__(self.uid, self.dof)

    def setup_control(self, mjData, mjModel, **kwargs) -> Tuple[dict[str, Any], dict[str, Any]]:
        actuators = {}
        joints = {}
        for name in self.joint_names:
            actuators[name] = mjData.actuator(name)
            joints[name] = mjData.joint(name)

        self.joints = joints
        self.actuators = actuators

        self.controller.set_initial_qpos(actuators, joints)

        if not self.joint_limits:
            for jname, j in self.joints.items():
                limits = mjModel.jnt_range[j.id]
                self.joint_limits[jname] = (limits[0], limits[1])

        return self.joints, self.actuators

    def apply_action(self, action_cmd) -> None:
        assert isinstance(self.controller, SingleArmBinaryEEFController)
        if action_cmd.type == "open_eef":
            self.controller.eef.open_gripper(self.actuators)
            return
        elif action_cmd.type == "close_eef":
            self.controller.eef.close_gripper(self.actuators)
            return
        else:
            target_qpos = action_cmd.parameters["target_qpos"]
            for jname, jval in target_qpos.items():
                if jname not in self.actuators:
                    continue
                self.actuators[jname].ctrl = jval

            if "eef_state" in action_cmd.parameters:
                eef_state = action_cmd.parameters["eef_state"]
                if eef_state == "open_eef":
                    self.controller.eef.open_gripper(self.actuators)
                elif eef_state == "close_eef":
                    self.controller.eef.close_gripper(self.actuators)

    def get_robot_qpos(self) -> dict[str, float]:
        """ Get the current joint positions of the robot. """
        return {j: v.qpos[0] for j, v in self.joints.items()}

    def get_grasp_pose_wrt_robot(self, grasp_info: dict, pregrasp: bool = False, robot_pose=None):
        assert robot_pose is not None, "not implemented"

        # Reused verbatim from Aloha.get_grasp_pose_wrt_robot: T_grasp_ee is a property of the
        # gripper-site frame convention (approach axis local +X, defined by the flip below), not
        # of which arm the site belongs to -- both of Aloha's gripper_site frames share the same
        # local convention, so the left-arm-only extraction changes nothing here.
        T_ee_hand = np.eye(4, dtype=np.float32)
        T_ee_hand[:3, 3] = np.array([-self.robot_eef_offset, 0, 0], dtype=np.float32)

        T_grasp_ee = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)  # only for graspnet

        R_world_grasp = t3d.quaternions.quat2mat(grasp_info['orientation'])

        T_world_grasp = np.eye(4, dtype=np.float32)
        # HACK: 0.01 offset to avoid collision (matches Aloha's own convention)
        T_world_grasp[:3, 3] = grasp_info['position'] + (grasp_info['depth'] - 0.01) * R_world_grasp[:, 0]
        T_world_grasp[:3, :3] = R_world_grasp

        if pregrasp:
            T_grasp_pregrasp = np.eye(4, dtype=np.float32)
            T_grasp_pregrasp[0, 3] = -np.random.uniform(self.pregrasp_distance[0], self.pregrasp_distance[1])
            T_pregrasp_robot_hand = np.linalg.inv(robot_pose) @ T_world_grasp @ T_grasp_pregrasp @ T_grasp_ee @ T_ee_hand
            grasp_pos_in_robot = T_pregrasp_robot_hand[:3, 3]
            grasp_ori_in_robot = t3d.quaternions.mat2quat(T_pregrasp_robot_hand[:3, :3])
        else:
            T_robot_hand = np.linalg.inv(robot_pose) @ T_world_grasp @ T_grasp_ee @ T_ee_hand
            grasp_pos_in_robot = T_robot_hand[:3, 3]
            grasp_ori_in_robot = t3d.quaternions.mat2quat(T_robot_hand[:3, :3])

        return (grasp_pos_in_robot, grasp_ori_in_robot)
