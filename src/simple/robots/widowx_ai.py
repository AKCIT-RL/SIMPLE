"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from typing import Any, List, Tuple

import numpy as np
import transforms3d as t3d

from simple.core.controller import Controller, ControllerCfg
from simple.core.robot import Robot
from simple.robots.controllers.combo import SingleArmBinaryEEFController, SingleArmBinaryEEFControllerCfg
from simple.robots.controllers.eef import ParallelGripperEEFController, ParallelGripperEEFControllerCfg
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.mixin import CuRoboMixin
from simple.robots.protocols import HasParallelGripper, WristCamMountable
from simple.robots.registry import RobotRegistry

# WidowX AI's gripper open/close travel (`left_carriage_joint`/`right_carriage_joint` range is
# "0 0.044" in wxai_follower.xml) is roughly double Franka's ("0 0.04"), and the trossen_arm_mujoco
# submodule's own diff-IK controller (third_party/trossen_arm_mujoco/.../src/controller.py) opens
# to 0.044 / closes to 0.022 -- not 0.0. `ParallelGripperEEFController.open_gripper/close_gripper`
# (src/simple/robots/controllers/eef.py) hardcodes 0.0205/0.0, which is Franka-specific, so we
# subclass rather than touch that shared file.
class WidowXParallelGripperEEFController(ParallelGripperEEFController):
    OPEN_CTRL: float = 0.044
    CLOSE_CTRL: float = 0.0

    def open_gripper(self, actuators: dict) -> None:
        for jname in self.cfg.joint_names:
            actuators[jname].ctrl = self.OPEN_CTRL

    def close_gripper(self, actuators: dict) -> None:
        for jname in self.cfg.joint_names:
            actuators[jname].ctrl = self.CLOSE_CTRL


class WidowXParallelGripperEEFControllerCfg(ParallelGripperEEFControllerCfg):
    clazz: type[Controller] = WidowXParallelGripperEEFController

    def __call__(self) -> WidowXParallelGripperEEFController:
        return WidowXParallelGripperEEFController(self, self.joint_names, self.init_qpos)


@RobotRegistry.register("widowx_ai")
class WidowXAI(CuRoboMixin, WristCamMountable, HasParallelGripper, Robot):
    uid: str = "widowx_ai"
    label: str = "Trossen WidowX AI"
    # 6 arm joints + left_carriage_joint (gripper actuator). right_carriage_joint (mujoco nq=8
    # includes it as an <equality>-mirrored mimic joint) is intentionally not counted here -- it
    # must match len(self.joints) / curobo cspace.joint_names (7), used by fk()/get_robot_qpos().
    dof: int = 7

    robot_cfg: Any = "robots/widowx_ai/curobo/widowx_ai.yml"
    mjcf_path: str = "robots/widowx_ai/wxai_follower.xml"
    # NOTE: no USD exists for WidowX AI anywhere (confirmed during integration research) -- this
    # path does not resolve to a real asset yet. Only sim_mode="mujoco" is supported until a USD
    # is authored/converted; IsaacSimEngine.add_robot() is not exercised by the mujoco-only tasks.
    usd_path: str = "robots/widowx_ai/widowx_ai.usd"

    robot_ns: str = "widowx_ai"
    hand_prim_path: str = "link_6"
    eef_prim_path: str = "ee_gripper_link"
    wrist_cam_link: str = "link_6"

    # Placeholder (identity) -- unused until a USD/IsaacSim path exists for this robot; the MuJoCo
    # wrist camera pose comes straight from wxai_follower.xml's own <camera name="cam" ...> element.
    wrist_camera_orientation: List[float] = [1.0, 0.0, 0.0, 0.0]

    # Offset from link_6 to ee_gripper_link, confirmed from the official URDF
    # (trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf):
    #   <joint name="ee_gripper" type="fixed"><origin xyz="0.156062 0 0"/>...</joint>
    # i.e. a pure translation along link_6's local +x axis (no rotation) -- matches the MJCF's
    # own `<site name="ee_site" pos="0.156 0 0"/>` inside link_6.
    robot_eef_offset: float = 0.156062
    pregrasp_distance: List[float] = [0.05, 0.08]
    visulize_spheres: bool = False

    joint_names: List[str] = [
        "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
        "left_carriage_joint", "right_carriage_joint",
    ]
    # right_carriage_joint is a <mimic joint="left_carriage_joint"/> in the URDF (and a MuJoCo
    # <equality> constraint in the MJCF) -- it is not an independent DOF, so it is intentionally
    # left out here (unlike Franka's two independently-actuated finger joints). This list's length
    # (7) must match curobo/widowx_ai.yml's `cspace.joint_names`.
    init_joint_states: dict[str, float] = {
        "joint_0": 0.0,
        "joint_1": 0.0,
        "joint_2": 0.0,
        "joint_3": 0.0,
        "joint_4": 0.0,
        "joint_5": 0.0,
        "left_carriage_joint": 0.044,  # gripper open by default, matching Franka's convention
    }

    controller_cfg: ControllerCfg = SingleArmBinaryEEFControllerCfg(
        arm=PDJointPosControllerCfg(
            joint_names=["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
            init_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        eef=WidowXParallelGripperEEFControllerCfg(
            joint_names=["left_carriage_joint"],
            init_qpos=[0.044],
        ),
    )

    def __init__(self):
        super().__init__(self.uid, self.dof)

    def setup_control(self, mjData, mjModel, **kwargs) -> Tuple[dict[str, Any], dict[str, Any]]:
        actuators = {}
        joints = {}
        for i in range(6):
            actuators[f"joint_{i}"] = mjData.actuator(f"joint_{i}")
            joints[f"joint_{i}"] = mjData.joint(f"joint_{i}")
        # Actuator is named "left_gripper" in wxai_follower.xml, but drives "left_carriage_joint".
        actuators["left_carriage_joint"] = mjData.actuator("left_gripper")
        joints["left_carriage_joint"] = mjData.joint("left_carriage_joint")
        # right_carriage_joint (the MJCF <equality>-mirrored gripper joint) is deliberately NOT
        # tracked here: self.joints must line up 1:1 with curobo's cspace.joint_names (7 DOF) for
        # CuRoboMixin.fk()/get_robot_qpos()-based observations to have matching lengths -- see
        # curobo/widowx_ai.yml's comment on the same exclusion.

        self.joints = joints
        self.actuators = actuators

        self.controller.set_initial_qpos(actuators, joints)

        # right_carriage_joint mimics left_carriage_joint exactly (MuJoCo <equality>,
        # polycoef="0 1 0 0 0" in wxai_follower.xml -- confirmed by reading the file directly),
        # but MuJoCo only resolves <equality> constraints via mj_step's constraint solver, never
        # via mj_forward -- and env.reset() only runs a handful of physics steps
        # (MujocoSimulator.step()), not always enough to converge the two carriages from a cold
        # start. Verified directly: right after reset, left_carriage_joint sat at 0.042 (open)
        # while right_carriage_joint was still at 0.020 -- visibly asymmetric gripper for however
        # long the episode takes to catch up (or the whole episode, if planning fails before any
        # open_eef/close_eef action ever gets applied). Setting it here, at the same instant
        # left_carriage_joint's own initial qpos is set, means both start already in sync instead
        # of relying on the solver to catch up.
        mjData.joint("right_carriage_joint").qpos[0] = joints["left_carriage_joint"].qpos[0]

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
                # Gripper is only ever driven via open_eef/close_eef/eef_state, never via raw
                # target_qpos (mirrors Franka's contract); MJCF/URDF joint names already match
                # 1:1 here (no "fr3_"/"panda_"-style prefix to strip, unlike Franka).
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

        # Both T_ee_hand and T_grasp_ee are IDENTITY for WidowX AI -- re-derived and numerically
        # validated (not reused from FrankaMixin), see
        # docs/teleop_simple_study/widowx_ai_grasp_pose_rederivation.md for the full derivation.
        # Short version: this function's whole point is converting a GraspNet grasp pose into
        # whatever pose curobo's own `ee_link` needs. For Franka, `ee_link: "fr3_hand"` (the
        # WRIST, per franka_fr3.yml) -- a separate frame from the actual gripper contact point
        # (`fr3_hand_tcp`), so T_ee_hand there does real work (steps back from the TCP to the
        # wrist by FRANKA_FINGER_LENGTH along -Z), and T_grasp_ee does real work too (fr3_hand's
        # own convention has its approach axis along Z, so GraspNet's approach axis, always local
        # X by convention, needs remapping X->-Z).
        # WidowX AI's curobo `ee_link` is `ee_gripper_link` (widowx_ai.yml) -- the gripper
        # contact point ITSELF, not a separate wrist frame -- so there is no "TCP -> wrist" step
        # to take at all (T_ee_hand = identity). And ee_gripper_link shares link_6's exact
        # orientation (confirmed directly from the URDF's fixed joint: <origin xyz="0.156062 0 0">
        # with no rpy, i.e. zero rotation), whose own approach axis is local +X (the direction
        # from link_6 out to the gripper, per robot_eef_offset) and whose finger-closing axis is
        # local Y (confirmed directly from wxai_follower.xml:
        # left_carriage_joint axis="0 1 0", right_carriage_joint axis="0 -1 0") -- i.e. WidowX's
        # own (approach=X, closing=Y) convention already matches GraspNet's own (approach=X,
        # binormal/closing=Y) convention exactly, so no rotation remapping is needed either
        # (T_grasp_ee = identity).
        # Verified numerically: a canonical straight-down GraspNet grasp (approach = world -Z),
        # run through this identity transform and then curobo FK, lands with ee_gripper_link's
        # own +X axis at world [0,0,-1] -- i.e. genuinely pointing straight down, matching the
        # intended approach direction exactly (not just "IK succeeds", the geometry is right).
        # 4 of 5 real cached GraspNet grasps for graspnet1b:0 became IK-reachable with this fix
        # (the previous FrankaMixin-copied matrices made the *canonical* straight-down grasp
        # IK-unreachable, not just an unlucky real one).
        T_ee_hand = np.eye(4, dtype=np.float32)

        T_grasp_ee = np.eye(4, dtype=np.float32)

        R_world_grasp = t3d.quaternions.quat2mat(grasp_info['orientation'])
        T_world_grasp = np.eye(4, dtype=np.float32)
        T_world_grasp[:3, 3] = grasp_info['position'] + grasp_info['depth'] * R_world_grasp[:, 0]
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

    def get_eef_pose_from_hand_pose(self, p, q):
        return p + t3d.quaternions.rotate_vector([self.robot_eef_offset, 0, 0], q), q

    def get_hand_pose_from_eef_pose(self, p, q, hack=0):
        EEF_TO_HAND = np.array([-(self.robot_eef_offset - hack), 0., 0.])
        p = p + t3d.quaternions.rotate_vector(EEF_TO_HAND, q)
        return p, q
