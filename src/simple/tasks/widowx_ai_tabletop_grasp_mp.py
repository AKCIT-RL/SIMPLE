"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional
if TYPE_CHECKING:
    from simple.core.randomizer import RandomizerCfg
from simple.core.task import Task
from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.scene import Scene
from simple.core.object import Object
from simple.core.actor import Actor, ObjectActor
from simple.core.layout import Layout
from simple.core.robot import Robot
from simple.dr import *
from simple.dr.manager import DRManager, TabletopGraspDRManager

from simple.dr.types import Box
from simple.robots.registry import RobotRegistry
from simple.tasks.registry import TaskRegistry
from simple.core.actor import ActorReigstry
from simple.sensors import StereoCameraCfg, SensorCfg, CameraCfg
from simple.robots.protocols import Controllable
from simple.robots.controllers.combo import SingleArmBinaryEEFControllerCfg
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.widowx_ai import WidowXParallelGripperEEFControllerCfg
from copy import deepcopy
from gymnasium import spaces
import numpy as np
from typing import Any
import math
from simple.assets import AssetManager

_LIFT_HEIGHT = 0.02

# Fixed robot spawn position (world xy) -- referenced both by `robot_region` below and by
# `_AnnularSectorBox`'s `center`, so the "radius from the robot base" the task asks for actually
# matches where the robot is placed instead of duplicating the same two numbers in two places.
_ROBOT_REGION = Box(low=[-0.3, -0.0], high=[-0.3, -0.0])

# Joint solution for the "close-high" Cartesian pose (p=[0.15, 0.00, 0.35], q=[1,0,0,0] wxyz,
# gripper open) from `scripts/test_widowx_ai_curobo_ik.py`'s WAYPOINTS list. Solved once via
# CuRobo IK with self-collision checking disabled (the WidowX AI self-collision default is known
# broken -- see docs/teleop_simple_study/widowx_ai_integration_status.md section 12) and
# independently cross-checked against MuJoCo's own hand-tuned collision primitives at this exact
# qpos (`mujoco.mj_forward` -> `d.ncon == 0`, genuinely collision-free, not just "curobo says so").
# FK residual at solve time was 4e-7 m. Used in place of WidowXAI's all-zero default init_qpos,
# which is the pose curobo's self-collision checker (incorrectly) flags as colliding at rest.
_CLOSE_HIGH_ARM_QPOS = [
    0.0,                    # joint_0
    0.13631485402584076,    # joint_1
    0.8843041062355042,     # joint_2
    -0.7479904890060425,    # joint_3
    0.0,                    # joint_4
    0.0,                    # joint_5
]


class _AnnularSectorBox(Box):
    """`Box` subclass that samples (x, y) from an angular sector of an annulus around `center`,
    instead of `Box`'s usual axis-aligned rectangle.

    `SpatialDR._random_place_one_object` (src/simple/dr/spatial.py:256) only ever calls
    `region.sample()` on `target_region`/`distractors_region` at runtime -- no `isinstance(Box)`
    check happens, so this only needs to look like a `Box` where it matters. Subclassing `Box`
    (rather than an unrelated class) keeps `RandomizerCfg.to_dict()` (src/simple/core/randomizer.py)
    working too -- it special-cases `isinstance(v, Box)` and pulls `.low`/`.high` for
    episode-metadata logging. `.low`/`.high` here are set to the sector's axis-aligned bounding
    box purely for that (lossy: center/radius/angle aren't captured by it), which is harmless --
    nothing reads that serialized dict back into a live region; reproducing a saved episode
    replays the *sampled* positions cached in `SpatialDR._inner_state`, not the region config.

    `half_angle` restricts sampling to a sector centered on `center`'s local +x axis (the
    direction the robot actually faces the table in this task -- `target_region` below was
    previously a rectangle entirely at positive x-offset from the robot for the same reason) --
    a full 0-2pi ring would risk placing objects behind the robot, off the table, or outside the
    WidowX AI's shorter reach (see the existing `spatial` comment on this task's `SpatialDRCfg`).
    """

    def __init__(self, center: list[float], r_low: float, r_high: float, half_angle: float):
        self.center = center
        self.r_low = r_low
        self.r_high = r_high
        self.half_angle = half_angle
        super().__init__(
            low=[center[0] + r_low * math.cos(half_angle), center[1] - r_high * math.sin(half_angle)],
            high=[center[0] + r_high, center[1] + r_high * math.sin(half_angle)],
        )

    def sample(self) -> list[float]:
        r = np.random.uniform(self.r_low, self.r_high)
        theta = np.random.uniform(-self.half_angle, self.half_angle)
        return [
            float(self.center[0] + r * math.cos(theta)),
            float(self.center[1] + r * math.sin(theta)),
        ]

@TaskRegistry.register("widowx_ai_tabletop_grasp_mp")
class WidowXAITabletopGraspTaskMP(Task):

    uid: str = "widowx_ai_tabletop_grasp_mp"
    label: str = "WidowX AI Tabletop Grasp Task"
    description: str = "A task that the WidowX AI robot must grasp target object on a tabletop."

    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "dr_level": 0,
        "version": 1.0,
    }

    robot_cfg: dict[str, Any] = dict(
        uid="widowx_ai",
    )

    sensor_cfgs: dict[str, SensorCfg] = dict(
        front_stereo = StereoCameraCfg(
            uid="Realsense_D415",
            mount="eye_on_base",
            width=640,
            height=360,
            focal_length=1.88,
            fov=np.deg2rad(71.28),
            near=0.2,
            far=5,
            baseline=0.055,
            pose=dict(
                # robot is at the origin, camera is orbiting around the robot
                distance=1.0,
                polar=np.deg2rad(60),
                azimuth=np.deg2rad(0),
            )
        ),
        # WidowX AI's wrist camera ("cam") is already a properly-mounted <camera> element inside
        # link_6 in wxai_follower.xml (a real RealSense D405, fovy=87 baked into the MJCF) --
        # mount="native" tells MujocoEngine._build_camera() to reuse it instead of creating a
        # (Franka-style, hardcoded-position, not actually attached) floating eye_in_hand camera.
        # The key here ("cam") MUST match the physical <camera name="cam"> element name.
        cam = CameraCfg(
            uid="Realsense_D405",
            mount="native",
            width=480,
            height=270,
            focal_length=1.88,
            fov=np.deg2rad(87),
            near=0.1,
            far=5,
            # Unused by mount="native" (the physical <camera> element's own pose in the MJCF
            # governs where it actually is) -- CameraEntity still requires a valid pose dict.
            pose=dict(position=[0.0, 0.0, 0.0], quaternion=[1.0, 0.0, 0.0, 0.0]),
        ),
    )

    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language = LanguageDRCfg(
            instructions = [
                "Pick up {}.",
            ]
        ),

        target = TargetDRCfg(
            asset_id="graspnet1b:0"
        ),

        distractors = DistractorDRCfg(
            res_id="objaverse",
            number_of_distractors=0,
            allow_duplicates=False,
            exclude=["0"]  # Exclude the target object
        ),

        # NOTE: WidowX AI has a shorter reach than the Franka FR3 (6-DOF small-form arm vs.
        # 7-DOF full-size arm). target_region is a 0.2-0.5m annular sector centered on the
        # robot's base (`_AnnularSectorBox` above) rather than an absolute-world rectangle, so
        # object placement stays defined relative to wherever the robot actually is. The
        # half_angle (35 deg) is a starting point copied from the general shape of the old
        # rectangular region, same caveat as before -- NOT independently re-validated for
        # WidowX's actual workspace; tune based on datagen IK-fail rate (same diagnostic used
        # during the Franka integration: a systematic 100%-episode IK_FAIL points at
        # spatial_region/collision setup, not a broken robot). distractors_region is untouched
        # (number_of_distractors=0 above means nothing spawns there today).
        spatial = SpatialDRCfg(
            spatial_mode="random",
            robot_region=_ROBOT_REGION,
            target_region=_AnnularSectorBox(
                center=_ROBOT_REGION.middle(),
                r_low=0.2,
                r_high=0.5,
                half_angle=math.radians(35),
            ),
            distractors_region=Box(low=[-0.2, -0.3], high=[1, 0.3]),
        ),
        camera = CameraDRCfg(
            cam_id="widowx_ai_camera",
        ),
        scene = TabletopSceneDRCfg(
            scene_mode="random",
            table_height=Box(low=0.0, high=0.0),
            scene_manager="hssd"
        ),
        lighting = LightingDRCfg(
            light_mode="random",
            light_num=(2,3),
            light_color_temperature=Box(low=4001, high=6001),
            light_intensity=Box(low=5e4, high=5e4),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 2.1),
            light_spacing=Box((1., 1.), (2.5, 2.5)),
            light_position=Box((-1.1, -1.1, 1.3), (1.1, 1.1, 1.5)),
            light_eulers=Box((0,0,-0.5*math.pi), (0,0,0.5*math.pi))
        ),

        material = MaterialDRCfg(
            material_mode="rand_all",
        )
    )

    def __init__(
        self,
        robot_uid: str = "widowx_ai",
        scene_uid: str | Scene | None = None,
        target_object: str | Object = "graspnet1b:0",
        split: str = "train",
        render_hz: int | None = None,
        dr_level: int = 0,
        physics_dt: float = 0.002,
        *args,
        **kwargs
    ):
        self._instruction = None
        self._target = None
        self._layout = None
        self._init_target_height = None

        self.robot_cfg.update(dict(
            uid=robot_uid,
        ))
        self._robot = RobotRegistry.make(**self.robot_cfg)

        # Spawn the robot at the "close-high" pose (`_CLOSE_HIGH_ARM_QPOS` above) instead of
        # WidowXAI's all-zero class default -- the all-zero pose is the one curobo's
        # self-collision checker (incorrectly) flags as colliding at rest (status doc section
        # 12); close-high is confirmed genuinely collision-free. `controller_cfg` is what
        # actually gets physically applied at reset (`PDJointPosController.set_initial_qpos`,
        # src/simple/robots/controllers/qpos.py:84 -- sets `mjData.joint(...).qpos` directly).
        # Setting it here as an instance attribute shadows the WidowXAI class attribute without
        # touching `robots/widowx_ai.py` -- `Controllable.controller`
        # (src/simple/robots/protocols.py) lazily builds `self.controller` from
        # `self.controller_cfg` on first access, which only happens later during
        # `setup_control()`, well after this constructor runs.
        self._robot.controller_cfg = SingleArmBinaryEEFControllerCfg(
            arm=PDJointPosControllerCfg(
                joint_names=["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
                init_qpos=_CLOSE_HIGH_ARM_QPOS,
            ),
            eef=WidowXParallelGripperEEFControllerCfg(
                joint_names=["left_carriage_joint"],
                init_qpos=[0.044],  # open, matching the close-high waypoint's gripper="open_eef"
            ),
        )

        # `robot.init_joint_states` ALSO needs to match close-high, separately from
        # `controller_cfg` above -- confirmed as a real, distinct bug (not redundant with the
        # override above): `controller_cfg` only controls the *physical* qpos MuJoCo starts at;
        # `init_joint_states` is a completely different dict that `CuRoboMixin
        # .planning_init_joint_states` (src/simple/robots/mixin.py:115) feeds to curobo as the
        # SEED configuration for the *first* planned trajectory (`CuRoboPlanner
        # .batch_plan_for_approach`, src/simple/mp/curobo.py:292) -- and it's also read directly
        # as a generic "nominal joint state" in several places in `agents/mp.py` and
        # `agents/base_agent.py`. Left at WidowXAI's all-zero class default, curobo planned the
        # first approach trajectory as if the arm were starting from all-zero, even though it
        # physically started at close-high -- so applying that trajectory's first waypoints
        # visibly snapped/teleported the arm from close-high back down toward all-zero over a
        # handful of steps before the real planned motion took over. Confirmed directly in a
        # recorded episode's `observation.joint_qpos`: frames 0-9 held close-high exactly, frames
        # 10-13 collapsed to near-all-zero while the recorded `action` stayed ~[0,0,0,0,0,0]
        # (i.e. not a real commanded motion -- a seed/reference mismatch, not deliberate
        # planning). See docs/teleop_simple_study/widowx_ai_init_joint_states_investigation.md.
        self._robot.init_joint_states = dict(
            zip(["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"], _CLOSE_HIGH_ARM_QPOS)
        ) | {"left_carriage_joint": 0.044}

        if scene_uid is not None:
            assert isinstance(self.dr_cfgs["scene"], TabletopSceneDRCfg)
            self.dr_cfgs["scene"].room_choices = [scene_uid] # type:ignore

        if target_object is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = target_object # type:ignore

            target_id = self.dr_cfgs["target"].asset_id.split(":")[-1]
            distractor_cfg = self.dr_cfgs.get("distractors")
            if distractor_cfg is not None and isinstance(distractor_cfg, DistractorDRCfg):
                if distractor_cfg.exclude is None:
                    distractor_cfg.exclude = []
                if target_id not in distractor_cfg.exclude:
                    distractor_cfg.exclude.append(target_id)

        drmgr = TabletopGraspDRManager(level=dr_level, **self.dr_cfgs)
        super().__init__(
            dr=drmgr,
            split=split,
            render_hz=render_hz,
            dr_level=dr_level,
            physics_dt=physics_dt,
            *args,
            **kwargs
        )

    @property
    def layout(self) -> Layout:
        """Returns the layout of the task."""
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def instruction(self) -> str:
        assert self._instruction is not None, "call reset() first"
        return self._instruction

    @property
    def target(self) -> Actor:
        assert self._target is not None, "call reset() first"
        return self._target

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    @property
    def observation_space(self) -> spaces.Space:
        default_obs = super().observation_space
        obs: dict[str, Any] = {
            "agent": spaces.Box(-np.pi, np.pi, shape=(self.robot.dof,), dtype=np.float32),
            "joint_qpos": spaces.Box(-np.pi, np.pi, shape=(self.robot.dof,), dtype=np.float32),
            "eef_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
        }
        if isinstance(default_obs, spaces.Dict):
            obs.update(dict(default_obs))
        return spaces.Dict(obs)

    def reset(self, seed:int|None=None, options: Optional[dict[str, Any]] = None) -> None:
        """Resets the task state."""
        super().reset(seed, options)
        split = self.metadata.get("split", "train")
        self._target = self.layout.actors.get("target")
        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        language_template = lang_dr(split)
        self._instruction = language_template.format(self._target.asset.name) # type: ignore
        self._init_target_height = None

    def clone_layout(self, options: dict[str, Any]) -> None:
        """Replicates an environment layout from given options."""
        self._layout = self.dr.replicate_env(self.robot, self.sensor_cfgs, deepcopy(options), self.split)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({})
        return state_dict

    def check_success(self,  info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info)
        return reward >= 1.0

    def compute_reward(self, info: dict[str, Any], **kwargs) -> float:
        target_obj_height = info["target"][2]
        if self._init_target_height is None:
            self._init_target_height = target_obj_height
        reward = np.clip((target_obj_height - self._init_target_height) / _LIFT_HEIGHT, 0, 1)
        return reward

    def preload_objects(self) -> list[Actor]:
        """Preloads all assets required by the task."""
        asset_manager = AssetManager.get("graspnet1b")
        return [ObjectActor(asset=asset) for asset in asset_manager]

    def decompose(self):
        from simple.datagen.subtask_spec import (
            OpenGripperSpec,
            CloseGripperSpec,
            GraspObjectSpec,
            LiftSpec
        )

        return [
            OpenGripperSpec("init"),
            GraspObjectSpec("approach", target_uid=self.target.uid, pregrasp=False),
            CloseGripperSpec("grasp"),
            LiftSpec("lift", up=0.1),
        ]
