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
from simple.robots.controllers.combo import (
    SingleArmGimbalBinaryEEFController,
    SingleArmGimbalBinaryEEFControllerCfg,
)
from simple.robots.controllers.eef import ParallelGripperEEFControllerCfg
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.mixin import CuRoboMixin
from simple.robots.protocols import HasParallelGripper, HeadCamMountable
from simple.robots.registry import RobotRegistry


# Named joint-position presets, for use as task spawn/reset configurations (mirroring
# `_CLOSE_HIGH_ARM_QPOS` in `widowx_ai_tabletop_grasp_mp.py`) and by
# `scripts/test_miss_gimbal_arm.py`. Arm/gripper values are Interbotix's own stock vx300s SRDF
# `group_state`s (`interbotix_xsarm_moveit/config/srdf/vx300s.srdf.xacro`) -- real, sourced
# values, not invented -- each cross-checked against the full Miss asset via
# `mujoco.mj_forward`/`d.ncon` for every arm x gripper x gimbal combination below.
#
# "sleep" is included for reference but is NOT safe to use as a spawn pose on Miss: folding the
# arm into Interbotix's stock Sleep configuration collides the gripper fingers with the chassis
# for every gripper/gimbal combination tried -- a real, pose-dependent collision specific to how
# the arm is mounted on this platform, unlike the pose-invariant "flush mount" contacts already
# excluded in miss.xml. "home" and "upright" are both confirmed collision-free (`d.ncon == 0`)
# for every gripper/gimbal combination tried.
#
# NOTE: the arm's own mount on the chassis was rotated 180deg about its own local Z in
# miss.xml/miss.urdf (see vx300s/shoulder_link's comment in miss.xml) to move the waist joint's
# +-pi wrap-around boundary away from "desk_reach" (which previously sat only ~0.0016 rad from the
# limit). "home"/"upright"/"sleep" below still use the SRDF-sourced `waist=0.0`/etc values, but
# `waist=0.0` now faces the opposite physical direction than it did before the mount rotation --
# not re-verified against the rotated mount (only "desk_reach" has been, see below).
ARM_PRESETS: dict[str, dict[str, float]] = {
    "home": dict(waist=0.0, shoulder=0.0, elbow=0.0, forearm_roll=0.0, wrist_angle=0.0, wrist_rotate=0.0),
    "upright": dict(waist=0.0, shoulder=0.0, elbow=-1.5708, forearm_roll=0.0, wrist_angle=0.0, wrist_rotate=0.0),
    "sleep": dict(waist=0.0, shoulder=-1.76, elbow=1.55, forearm_roll=0.0, wrist_angle=0.8, wrist_rotate=0.0),  # NOT collision-free on Miss, see above
    # User-chosen default (see Miss.init_joint_states) -- re-derived after the arm-mount rotation
    # (waist=0.0 now sits far from the +-pi limit, unlike the pre-rotation -3.14 value), confirmed
    # collision-free for the full Miss asset (mujoco.mj_forward -> d.ncon==0) and for curobo's own
    # self-collision IKSolver query at this exact configuration.
    "desk_reach": dict(waist=0.0, shoulder=-1.60, elbow=0.20, forearm_roll=-0.0, wrist_angle=1.20, wrist_rotate=-0.0),
}

# Gripper presets are expressed as left_finger only -- right_finger mirrors it via miss.xml's
# equality constraint / miss.urdf's <mimic> (see Miss.setup_control's docstring).
GRIPPER_PRESETS: dict[str, float] = {
    "grasping": 0.021,
    "released": 0.057,
    "home": 0.03,
}

# No equivalent presets exist anywhere in the misskal_hardware repo for the gimbal (only its raw
# joint limits) -- these are authored fresh for SIMPLE, not sourced. "stow" is the class-level
# default (gimbal_init_qpos). The camera-pointing presets are a first guess at what "look toward
# a tabletop workspace" means for this mount -- combined with gimbal_cam's own unverified
# orientation (see miss.xml), treat these as a starting point to iterate on visually via
# scripts/test_miss_gimbal_arm.py, not as validated values.
GIMBAL_PRESETS: dict[str, dict[str, float]] = {
    "stow": dict(head_pan=0.0, head_tilt=0.0),
    "look_down": dict(head_pan=0.0, head_tilt=0.8),
    "look_down_more": dict(head_pan=0.0, head_tilt=1.5),
    # User-chosen default (see Miss.gimbal_init_qpos) -- pairs with the "desk_reach" arm preset.
    "look_at_desk": dict(head_pan=0.0, head_tilt=-1.9025),
}


@RobotRegistry.register("miss")
class Miss(CuRoboMixin, HeadCamMountable, HasParallelGripper, Robot):
    """The user's real "Miss" platform: a Clearpath Jackal (j100) wheeled base carrying a
    single ViperX arm (stock Interbotix vx300s, not Aloha's fork -- see
    docs/source/miss/miss_integration.md) and a 2-DOF pan/tilt gimbal with an Intel RealSense
    D455.

    `data/robots/miss/miss.xml` is the FULL platform geometry (chassis, wheels, lidar, the
    second static d435i camera, the arm, the stock gripper, and the gimbal) -- mechanically
    flattened from the real `viperx/misskal_hardware` xacro sources (see the module docstring in
    `data/robots/miss/miss.urdf`'s generation script), not hand-authored. Only the arm, gripper,
    and gimbal are functional here: `base_link` has no root joint (rigidly attached to the
    world, same as every other manipulator robot in SIMPLE today), and the four wheel joints
    exist in the model (so the asset is genuinely complete) but are never actuated.

    TODO (deliberately out of scope for now, see docs/source/miss/miss_integration.md): give
    `base_link` a real mobile joint, actuate the wheels, and build whatever SIMPLE
    engine/task-family support a driven wheeled base needs. No existing SIMPLE robot or task
    does this today.
    """

    uid: str = "miss"
    label: str = "Miss (Jackal j100 + ViperX + gimbal)"
    # 6 arm joints + left_finger (right_finger mirrors it via an <equality> constraint in
    # miss.xml -- MuJoCo's URDF importer does not honor URDF <mimic>, confirmed directly against
    # the raw xacro-flattened output, so this equality was authored by hand, same convention as
    # WidowX AI's carriage-joint exclusion). Matches curobo/miss.yml's cspace.joint_names length.
    dof: int = 7

    robot_cfg: Any = "robots/miss/curobo/miss.yml"
    mjcf_path: str = "robots/miss/miss.xml"
    # No USD exists (or is planned) for this platform -- placeholder for API symmetry only,
    # matching WidowXAI's/ViperX's precedent. Only sim_mode="mujoco" is supported.
    usd_path: str = "robots/miss/miss.usd"

    robot_ns: str = "miss"
    hand_prim_path: str = "vx300s/gripper_bar_link"
    eef_prim_path: str = "vx300s/ee_gripper_link"

    # MujocoEngine's synthetic checkered "groundplane" (the only floor-like geometry that exists
    # in the MuJoCo backend at all -- there is no HSSD room mesh outside IsaacSim) is normally
    # placed relative to the *table*'s own height (`z_minus = table.pose.position[2] + 0.5 *
    # table.size[2]`), which makes sense for a robot bolted to the tabletop (Franka/Aloha/ViperX/
    # WidowX AI all shift their own root down by -table_height so the table's fixed floor-level
    # placement reads as "the right height" relative to them) but is simply wrong for Miss: its
    # wheels touch world Z=0 regardless of table_height, and that formula's table-height-dependent
    # term has no relationship to Miss's own floor contact at all (confirmed directly: at
    # table_height=0.70 the groundplane landed 70cm below the wheels -- the "floating" the user
    # reported). Vega1's own `z_offset` attribute (`src/simple/robots/vega.py`) only shifts the
    # groundplane by a *constant*, which cannot fix this (Miss's error term varies with
    # table_height, a per-task-instance knob) -- floor_grounded is a new, separate opt-in that
    # tells the engine to ignore the table term entirely and place the groundplane at exactly this
    # robot's own known Z instead. See `MujocoEngine`'s groundplane-placement code for the other
    # side of this.
    floor_grounded: bool = True

    # The gimbal-mounted D455 (see the camera element and its own TODOs in miss.xml), not a
    # wrist camera -- Miss's arm carries no camera, unlike Aloha's/ViperX's fork.
    head_cam_link: str = "gimbal_head"
    head_camera_orientation: List[float] = [1.0, 0.0, 0.0, 0.0]  # placeholder, see miss.xml

    robot_eef_offset: float = 0.0385  # fingers_link -> ee_gripper_link, from vx300s.urdf.xacro
    pregrasp_distance: List[float] = [0.05, 0.08]
    visulize_spheres: bool = False

    joint_names: List[str] = [
        "waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "left_finger",
    ]
    # User-chosen "desk reach" default pose (arm/gripper/gimbal). Also registered as
    # ARM_PRESETS["desk_reach"] / GIMBAL_PRESETS["look_at_desk"] below.
    #
    # Re-derived after the arm-mount rotation (see vx300s/shoulder_link's comment in miss.xml):
    # `waist` moved from -3.14 (only ~0.0016 rad from the +-pi joint-range limit, suspected of
    # biasing curobo's motion planning toward long "the way around" waist solutions) to 0.0, now
    # far from either limit. Confirmed collision-free for the full Miss asset directly
    # (mujoco.mj_forward -> d.ncon==0) and for curobo's own self-collision IKSolver query at this
    # exact configuration -- both checked before adopting this as the default, the same way
    # _CLOSE_HIGH_ARM_QPOS was checked for WidowX AI. Must stay in sync with
    # `controller_cfg.arm.init_qpos` below and `ARM_PRESETS["desk_reach"]` above.
    init_joint_states: dict[str, float] = {
        "waist": 0.0,
        "shoulder": -1.60,
        "elbow": 0.20,
        "forearm_roll": -0.0,
        "wrist_angle": 1.20,
        "wrist_rotate": -0.0,
        "left_finger": 0.057,  # open
    }

    # The gimbal is not part of the arm's kinematic chain (mounted on the base, not the wrist)
    # and is deliberately kept out of joint_names/dof/init_joint_states above -- those three stay
    # exactly curobo-cspace-sized (7), matching the WidowX AI/ViperX convention that
    # CuRoboMixin.fk()/planning_init_joint_states() rely on. The gimbal gets its own parallel
    # bookkeeping instead, wired into the same self.joints/self.actuators dict by
    # setup_control() below (a plain name->object map, not size-constrained to joint_names).
    gimbal_joint_names: List[str] = ["head_pan", "head_tilt"]
    # [head_pan, head_tilt] = [0.0, -1.9025] -- the user specified this as gimbal=[-1.9025, 0.0]
    # using the [pitch, yaw] convention established by miss_interactive_control.py/
    # test_miss_curobo_ik.py (pitch->head_tilt, yaw->head_pan); transposed here to match this
    # list's own [head_pan, head_tilt] order. Looking down toward the desk, per "look_at_desk".
    gimbal_init_qpos: List[float] = [0.0, -1.9025]

    controller_cfg: ControllerCfg = SingleArmGimbalBinaryEEFControllerCfg(
        arm=PDJointPosControllerCfg(
            joint_names=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
            init_qpos=[0.0, -1.60, 0.20, -0.0, 1.20, -0.0],
        ),
        eef=ParallelGripperEEFControllerCfg(
            joint_names=["left_finger"],
            init_qpos=[0.057],
            # left_finger's actuator ctrlrange is "0.021 0.057" (miss.xml) -- the base class's
            # default open/close values (0.0205/0.0) both fall outside this range and clamp to the
            # same boundary, so open_gripper()/close_gripper() would otherwise be indistinguishable
            # (see the comment in ParallelGripperEEFController.open_gripper).
            open_qpos=[0.057],
            close_qpos=[0.021],
        ),
        gimbal=PDJointPosControllerCfg(
            joint_names=["head_pan", "head_tilt"],
            init_qpos=[0.0, -1.9025],
        ),
    )

    def __init__(self):
        super().__init__(self.uid, self.dof)

    def setup_control(self, mjData, mjModel, **kwargs) -> Tuple[dict[str, Any], dict[str, Any]]:
        actuators = {}
        joints = {}
        for name in self.joint_names + self.gimbal_joint_names:
            actuators[name] = mjData.actuator(name)
            joints[name] = mjData.joint(name)
        # right_finger is equality-constrained (mimics left_finger, see miss.xml) rather than an
        # independently actuated joint, matching how WidowX AI's setup_control does NOT track
        # right_carriage_joint here either -- kept out of self.joints/self.actuators, since
        # nothing needs to command it directly.

        self.joints = joints
        self.actuators = actuators

        self.controller.set_initial_qpos(actuators, joints)
        # right_finger has no actuator of its own (see the comment above) and is left out of
        # self.joints, so nothing else ever sets its qpos -- it would otherwise start each episode
        # at MuJoCo's implicit default of 0.0, which is OUTSIDE right_finger's own joint range
        # ([-0.057, -0.021], mirrored from left_finger's [0.021, 0.057]) and inconsistent with the
        # miss.xml equality constraint (`right_finger = -left_finger`) for whatever left_finger was
        # just initialized to. The compliant equality solver then has to fight both its own
        # out-of-range joint limit and the equality pull at once, with no actuator to help --
        # visible as one gripper finger snapping to target while the other lags/sits wrong,
        # especially right after reset. Mirroring qpos here (matching the xacro's dropped <mimic
        # multiplier="-1">, restored by hand as the miss.xml equality) starts it in a physically
        # consistent state instead.
        right_finger = mjData.joint("right_finger")
        right_finger.qpos = -joints["left_finger"].qpos
        right_finger.qvel = 0
        right_finger.qacc = 0
        for jname, val in zip(self.gimbal_joint_names, self.gimbal_init_qpos):
            joints[jname].qpos = val
            joints[jname].qvel = 0
            actuators[jname].ctrl = val

        if not self.joint_limits:
            for jname, j in self.joints.items():
                limits = mjModel.jnt_range[j.id]
                self.joint_limits[jname] = (limits[0], limits[1])

        return self.joints, self.actuators

    def apply_action(self, action_cmd) -> None:
        assert isinstance(self.controller, SingleArmGimbalBinaryEEFController)
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

            if "gimbal_qpos" in action_cmd.parameters and action_cmd.parameters["gimbal_qpos"] is not None:
                for jname, jval in action_cmd.parameters["gimbal_qpos"].items():
                    self.actuators[jname].ctrl = jval

            if "eef_state" in action_cmd.parameters:
                eef_state = action_cmd.parameters["eef_state"]
                if eef_state == "open_eef":
                    self.controller.eef.open_gripper(self.actuators)
                elif eef_state == "close_eef":
                    self.controller.eef.close_gripper(self.actuators)

    def get_robot_qpos(self) -> dict[str, float]:
        """ Get the current joint positions of the robot (arm+gripper only, matching dof/
        joint_names -- see the class docstring for why the gimbal is tracked separately). """
        return {j: self.joints[j].qpos[0] for j in self.joint_names}

    def get_gimbal_qpos(self) -> dict[str, float]:
        return {j: self.joints[j].qpos[0] for j in self.gimbal_joint_names}

    def get_grasp_pose_wrt_robot(self, grasp_info: dict, pregrasp: bool = False, robot_pose=None):
        assert robot_pose is not None, "not implemented"

        # TODO: unlike ViperX (which reused Aloha's *_gripper_site frame convention verbatim,
        # already validated for the identical hardware), Miss's stock Interbotix gripper has no
        # equivalent *_gripper_site frame or prior grasp-pose derivation to reuse -- its
        # ee_gripper_link frame convention (specifically whether GraspNet's local+X/local+Y
        # approach/closing axes line up with ee_gripper_link's own local axes) has not been
        # independently verified against GraspNet's convention the way WidowX AI's was (see
        # widowx_ai.py's own get_grasp_pose_wrt_robot for the derivation this needs to repeat) --
        # so T_grasp_ee (rotation only) is still left as identity, do not trust this for real
        # grasp planning without first verifying it the same way WidowX AI's was.
        #
        # T_ee_hand's translation, however, was a confirmed, reproducible bug on its own, fixed
        # here: `ee_gripper_link` is curobo's IK target frame, but it is NOT where the fingers
        # close -- that's `robot_eef_offset` (0.0385m) further along the approach axis, a value
        # this function already had available but never applied. Leaving it as identity commanded
        # curobo to put ee_gripper_link itself at the object's grasp contact point, i.e. ~3.85cm
        # too far forward along the approach direction -- confirmed directly: with identity, real
        # `datagen` grasp candidates landed ~1.4cm above table2's real top surface (right at/into
        # it), producing `MotionGenStatus.IK_FAIL` on every attempt despite the pose being
        # trivially IK-reachable in isolation (no-collision `robot.ik()` succeeded for the exact
        # same poses) -- i.e. this was a real collision against the desk, not an unreachable pose.
        # Applying the offset (mirroring ViperX/Aloha's own `-robot_eef_offset` along local X)
        # pulls the commanded wrist back by 3.85cm, moving it to ~4cm clearance above the table
        # instead -- reproduced fixed for multiple real grasp candidates before landing this.
        T_ee_hand = np.eye(4, dtype=np.float32)
        T_ee_hand[:3, 3] = np.array([-self.robot_eef_offset, 0, 0], dtype=np.float32)
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
