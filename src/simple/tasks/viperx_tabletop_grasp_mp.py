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
from copy import deepcopy
from gymnasium import spaces
import numpy as np
from typing import Any
import math
from simple.assets import AssetManager

_LIFT_HEIGHT = 0.02


@TaskRegistry.register("viperx_tabletop_grasp_mp")
class ViperXTabletopGraspTaskMP(Task):
    """Single-arm ViperX tabletop grasp task.

    ViperX (`src/simple/robots/viperx.py`) is Aloha's left arm re-exposed as its own single-arm
    embodiment (see docs/source/viperx/viperx_integration.md). This task mirrors
    `widowx_ai_tabletop_grasp_mp.py`'s structure (a genuinely single-arm task, unlike
    `aloha_tabletop_grasp_mp.py`, whose `decompose()`/camera set assume a bimanual robot), but
    reuses Aloha's own sensor/spatial-DR configuration verbatim -- the underlying MJCF geometry
    for the left arm and table is unchanged from Aloha's own (already-tuned) tabletop grasp task.
    """

    uid: str = "viperx_tabletop_grasp_mp"
    label: str = "ViperX Tabletop Grasp Task"
    description: str = "A task where the single-arm ViperX robot must grasp a target object on a tabletop."

    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "dr_level": 0,
        "version": 1.0,
    }

    robot_cfg: dict[str, Any] = dict(
        uid="viperx",
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
                distance=1.35,
                polar=np.deg2rad(60),
                azimuth=np.deg2rad(0),
            )
        ),
        # Aloha's own left-arm wrist camera config (src/simple/tasks/aloha_tabletop_grasp_mp.py) --
        # reused unchanged rather than switching to mount="native" against viperx.xml's own
        # "wrist_cam_left" element, to keep this task's sensor behavior identical to the
        # already-validated Aloha production task it was extracted from.
        wrist = CameraCfg(
            uid="Logitech_C930e",
            mount="eye_in_hand",
            width=480,
            height=270,
            focal_length=0.94,
            fov=np.deg2rad(90),
            near=0.1,
            far=5,
            pose=dict(
                position=[0.0, 0.0, 0.0],
            )
        ),
    )

    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language = LanguageDRCfg(
            instructions = [
                "Pick up {}.",
            ]
        ),

        target = TargetDRCfg(
            asset_id="graspnet1b:63"
        ),

        distractors = DistractorDRCfg(
            res_id="graspnet1b",
            number_of_distractors=3,
            allow_duplicates=False,
            exclude=["63"]  # Exclude the target object
        ),

        # Reused verbatim from aloha_tabletop_grasp_mp.py -- the pruned viperx.xml keeps the left
        # arm at the exact same offset from the world/curobo base_link frame as Aloha's own
        # left_base_link, so the same table/robot/target geometry that was already tuned for
        # Aloha's left arm applies unchanged here.
        spatial = SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[0.1, -0.0], high=[0.1, -0.0]),
            target_region=Box(low=[0.1, -0.1], high=[0.1, 0.1]),
            distractors_region=Box(low=[-0.3, -0.3], high=[0.3, 0.3]),
        ),

        camera = CameraDRCfg(
            cam_id="viperx_camera",
        ),

        scene = TabletopSceneDRCfg(
            table_height=Box(low=0.0, high=0.0),
            scene_manager="hssd"
        ),

        lighting = LightingDRCfg(
            light_mode="random",
            light_num=(2,3),
            light_color_temperature=Box(low=4001, high=6001),
            light_intensity=Box(low=8e4, high=1e5),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 2.1),
            light_spacing=Box((1., 1.), (2.5, 2.5)),
            light_position=Box((-1.1, -1.1, 2.1), (1.1, 1.1, 4.1)),
            light_eulers=Box((0,0,-0.5*math.pi), (0,0,0.5*math.pi))
        ),

        material = MaterialDRCfg(
            material_mode="rand_all",
        )
    )

    def __init__(
        self,
        robot_uid: str = "viperx",
        target_object: str | None = "graspnet1b:63",
        scene_uid: str | None = None,
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

        if scene_uid is not None:
            assert isinstance(self.dr_cfgs["scene"], TabletopSceneDRCfg)
            self.dr_cfgs["scene"].room_choices = [scene_uid] # type:ignore

        if target_object is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = target_object

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

        # No hand_uid: ViperX has exactly one arm, unlike Aloha's decompose() which pins
        # hand_uid="left" throughout. subtask_spec.py/agents/mp.py already treat hand_uid as an
        # optional dict key, so simply omitting it is sufficient here.
        return [
            OpenGripperSpec("init"),
            GraspObjectSpec("approach", target_uid=self.target.uid, pregrasp=False),
            CloseGripperSpec("grasp"),
            LiftSpec("lift", up=0.1),
        ]
