"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from simple.core.randomizer import RandomizerCfg
    from simple.datagen.subtask_spec import SubtaskSpec

from typing import Any

import numpy as np
from gymnasium import spaces

from simple.assets import AssetManager
from simple.core.actor import Actor, ActorReigstry, ObjectActor
from simple.core.layout import Layout
from simple.core.object import Object
from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.robot import Robot
from simple.core.scene import Scene
from simple.core.task import Task
from simple.dr import *
from simple.dr.manager import DRManager, TabletopGraspDRManager 
from simple.dr.types import Box
from simple.robots.protocols import Controllable
from simple.robots.registry import RobotRegistry
from simple.sensors import CameraCfg, SensorCfg, StereoCameraCfg
from simple.tasks.registry import TaskRegistry

_PLACE_HEIGHT = 0.5
_LOWER_HEIGHT = 0.1

# Used only when industrial_material=True (see __init__): pins both table and
# table2 to the same industrial metallic look instead of the default per-episode
# random material.
_INDUSTRIAL_METAL_MATERIAL = {
    "path": "vMaterials_2/Metal/Metal_Cast.mdl",
    "name": "Metal_Cast",
}


@TaskRegistry.register("g1_industrial_screw_to_tote_teleop")
class G1IndustrialScrewToToteTeleop(Task):
    uid: str = "g1_industrial_screw_to_tote_teleop"
    label: str = "G1 TELEOP Industrial Screw to Tote to Workbench"
    description: str = (
        "An industrial task where the G1 robot must pick up a screw from the assembly table, "
        "place it into a tote, and carry the tote to the dispatch workbench."
    )

    metadata: dict[str, Any] = {
        "physics_dt": 0.005,
        "control_hz": 200,
        "render_hz": 50,
        "dr_level": 0,
        "version": 1.0,
        "reward_dt": 0.02,
        "image_dt": 0.033333,
        "need_gravity": True,
        "max_episode_steps": 1200,
    }

    robot_cfg: dict[str, Any] = dict(
        uid="g1_sonic",
    )

    sensor_cfgs: dict[str, SensorCfg] = dict(
        head_stereo=StereoCameraCfg(
            uid="Realsense_D435i",
            mount="eye_in_head",
            width=640,
            height=360,
            focal_length=1.93,
            fov=np.deg2rad(110),
            near=0.2,
            far=5,
            baseline=0.05,
            pose=dict(
                position=[0.0, 0.0, 0.0],
            ),
        ),
    )

    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language=LanguageDRCfg(
            instructions=[
                "pick up the screw from the assembly table, put it in the tote, and move the tote to the dispatch workbench.",
            ]
        ),
        target=TargetDRCfg(asset_id="graspnet1b:27"),  # "metal screw"
        container=TargetDRCfg(asset_id="totes:bin_b04"),
        distractors=DistractorDRCfg(
            res_id="graspnet1b",
            number_of_distractors=4,
            allow_duplicates=False,
            exclude=["0", "12", "46", "6", "27"],
        ),
        spatial=SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[-0.63, 0, 0], high=[-0.65, 0, 0]),
            target_region=Box(low=[-0.20, -0.04], high=[-0.22, 0.04]),
            container_region=Box(low=[-0.20, 0.60], high=[-0.22, 0.65]), 
            container_rotate_z=Box(low=3.1415, high=3.1415),
            distractors_region=[
                Box(low=[-0.2, -0.3], high=[-0.0, 0.3]),
                Box(low=[-0.2, -0.3], high=[-0.0, 0.3]),
                Box(low=[-0.2, -0.3], high=[-0.0, 0.3]),
                Box(low=[-2.8, -0.3], high=[-2.6, 0.3]), 
            ],
            target_stable_indices=[0],
            target_rotate_z=Box(low=np.pi, high=np.pi),
            obj_surface_map={
                "target": "table",
                "container": "table", 
                "distractor": ["table", "table", "table", "table2"],
            },
        ),
        camera=CameraDRCfg(
            cam_id="franka_camera",
        ),
        scene=TabletopSceneDRCfg(
            scene_manager="warehouse", 
            room_choices=["warehouse:default"],
            scene_mode="fixed",
            table_size=Box(low=[0.8, 2.0, 0.1], high=[0.8, 2.0, 0.1]),
            table_position=Box(low=[0, 0.25], high=[0, 0.25]),
            table_height=Box(low=0.75, high=0.75), 
            rotation_z=Box(low=0.0, high=0.0),
            table2_size=Box(low=[0.8, 2.0, 0.1], high=[0.8, 2.0, 0.1]),
            table2_position=Box(low=[-2.5, 0.25], high=[-2.5, 0.25]), 
            table2_height=Box(low=0.75, high=0.75), 
            table2_rotation_z=Box(low=0.0, high=0.0),
            enable_table2=True,
        ),
        lighting=LightingDRCfg(
            light_mode="random", 
            light_num=(2, 3),
            light_color_temperature=Box(low=6001, high=8001), 
            light_intensity=Box(low=5e4, high=5e4),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 2.1),
            light_spacing=Box((1.0, 1.0), (2.5, 2.5)),
            light_position=Box((-1.1, -1.1, 1.3), (1.1, 1.1, 1.5)),
            light_eulers=Box((0, 0, -0.5 * np.pi), (0, 0, 0.5 * np.pi)),
        ),
        material=MaterialDRCfg(
            # Default: table and table2 are both randomized every episode, same
            # as the rest of the pipeline. Pass industrial_material=True to pin
            # both to a fixed industrial metallic look instead (see __init__).
            material_mode="rand_all",
        ),
    )

    def __init__(
        self,
        robot_uid: str = "g1_sonic",
        scene_uid: str | Scene = "warehouse:default",
        target_object: str | Object = "graspnet1b:27",
        controller_uid: str = "pd_joint_pos",
        split: str = "train",
        render_hz: int | None = None,
        dr_level: int = 0,
        success_criteria: float = 0.9,
        industrial_material: bool = False,
        *args,
        **kwargs,
    ):
        self._instruction = None
        self._target = None
        self._container = None
        self._layout = None
        self._init_target_height = None
        self._contact_started = False

        self.robot_cfg.update(
            dict(
                uid=robot_uid,
            )
        )

        self.reward = 0
        self.success_criteria = success_criteria

        self._robot = RobotRegistry.make(**self.robot_cfg, **kwargs)

        if target_object is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = target_object  # type:ignore

            target_id = self.dr_cfgs["target"].asset_id.split(":")[-1]
            distractor_cfg = self.dr_cfgs.get("distractors")
            if distractor_cfg is not None and isinstance(
                distractor_cfg, DistractorDRCfg
            ):
                if distractor_cfg.exclude is None:
                    distractor_cfg.exclude = []
                if target_id not in distractor_cfg.exclude:
                    distractor_cfg.exclude.append(target_id)

        if industrial_material:
            material_cfg = self.dr_cfgs.get("material")
            assert isinstance(material_cfg, MaterialDRCfg)
            # Pin table and table2 to the same metallic look together; a single
            # table randomized against a fixed other one wouldn't make sense.
            material_cfg.fixed_table_material = _INDUSTRIAL_METAL_MATERIAL
            material_cfg.fixed_table2_material = _INDUSTRIAL_METAL_MATERIAL

        drmgr = TabletopGraspDRManager(level=dr_level, **self.dr_cfgs)
        super().__init__(
            dr=drmgr,
            split=split,
            render_hz=render_hz,
            dr_level=dr_level,
            *args,
            **kwargs,
        )

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def instruction(self) -> str:
        assert self._instruction is not None, "call reset() first"
        return self._instruction  # type: ignore

    @property
    def target(self) -> Actor:
        assert self._target is not None, "call reset() first"
        return self._target

    @property
    def container(self) -> Actor:
        assert self._container is not None, "call reset() first"
        return self._container

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    @property
    def observation_space(self) -> spaces.Space:
        default_obs = super().observation_space
        obs: dict[str, Any] = {
            "joint_qpos": spaces.Box(
                -np.pi, np.pi, shape=(self.robot.wholebody_dof,), dtype=np.float32
            ),  # type:ignore
        }
        if isinstance(default_obs, spaces.Dict):
            obs.update(dict(default_obs))
        return spaces.Dict(obs)

    def reset(
        self, seed: int | None = None, options: Optional[dict[str, Any]] = None
    ) -> None:
        super().reset(seed, options)
        split = self.metadata.get("split", "train")
        self._target = self.layout.actors.get("target")
        self._container = self.layout.actors.get("container")
        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        language_template = lang_dr(split)
        self._instruction = language_template.format(self._target.asset.name)  # type: ignore
        self._init_target_height = None
        self._contact_started = False
        self.reward = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update(
            {
                "container_uid": self.container.uid if self._container else None,
            }
        )  
        return state_dict

    def check_container_on_table2(self, *args, **kwargs) -> bool:
        """
        Check if the tote is on table2 (dispatch workbench).
        """
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False

        container_name = str(self.container.asset.label)

        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name

            if (container_name in body1 and "table2" in body2) or (
                container_name in body2 and "table2" in body1
            ):
                return True
        return False

    def check_object_in_container(self, *args, **kwargs) -> bool:
        """
        Check if the screw is successfully placed in the tote.
        """
        container_pos = np.array(self.layout.actors["container"].pose.position)
        target_pos = np.array(self.layout.actors["target"].pose.position)

        err_xy = np.abs(container_pos[:2] - target_pos[:2])
        success = np.all(err_xy <= 0.12)

        target_name = str(self.target.asset.label)
        container_name = str(self.container.asset.label)
        mujoco_env = kwargs.get("mujoco_env", None)

        if mujoco_env is None:
            return False

        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name

            if (target_name in body1 and container_name in body2) or \
               (target_name in body2 and container_name in body1) and success:
                return True
        return False

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info, *args, **kwargs)
        return reward >= self.success_criteria

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        is_object_in_container = self.check_object_in_container(info=info, *args, **kwargs)
        is_container_on_dispatch = self.check_container_on_table2(*args, **kwargs)

        if is_object_in_container and is_container_on_dispatch:
            self.reward = 1.0
        elif is_object_in_container:
            self.reward = 0.5
        else:
            self.reward = 0.0
        return self.reward

    def preload_objects(self) -> list[Actor]:
        """Preloads all assets required by the task."""
        asset_manager = AssetManager.get("graspnet1b")
        return [ObjectActor(asset=asset) for asset in asset_manager]
