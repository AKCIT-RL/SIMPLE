"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

G1 (fixed/rubber hand) tabletop COLOUR-BOX SORTING teleop task.

Three solid-colour cubes (red / green / blue) start scrambled on a bench within
arm reach; the robot must PUSH each cube onto the flat colour pad of the SAME
colour. The rubber hand has no fingers (see robots/g1_sonic_fixedhand.py), so the
skill is a push, not a grasp -- which is exactly why the targets are flat pads
(easy to push a cube across) rather than totes (hard to lift over a rim).

Control: the standard ``decoupled_wbc`` locomotion stack (same as every other
sonic teleop task) -- the robot stands in front of the bench and does not need to
walk. There is intentionally no fixed-base mode: the WBC needs the floating base,
and a welded root is disallowed by the stack (g1_sonic.py setup_control).

Scene: a single PRIMITIVE table (TabletopSceneDR) -- no warehouse furniture, no
operator-frame remap. The robot spawns at the canonical teleop-safe pose (-x
side, yaw 0, facing +x toward the bench); the cubes and pads are authored
directly in that canonical frame, in front of the robot.

Assets:
  * cubes  -> boxes:box_{red,green,blue}   (ObjectActor, movable, rgba-tinted;
              assets/boxes/cube.obj + cube.usda)
  * pads   -> static 0-joint articulated markers, visual only (no collision), so
              cubes slide across them (assets/boxes/pads/pad_*.xml + .usda)

Success (episode completion): 1/3 per cube resting on its matching-colour pad
(cube centre within the pad footprint AND on the tabletop). success_criteria 0.9
=> the episode only completes when all three cubes are sorted. Checks use live
MuJoCo body positions, not the initial layout poses.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import transforms3d as t3d
from gymnasium import spaces

from simple.assets import AssetManager
from simple.core.actor import Actor, ObjectActor
from simple.core.asset import ArticulatedAsset
from simple.core.layout import Layout
from simple.core.object import Object
from simple.core.randomizer import RandomizerCfg
from simple.core.scene import Scene
from simple.core.task import Task
from simple.dr import *
from simple.dr.manager import TabletopGraspDRManager
from simple.dr.types import Box
from simple.robots.protocols import Controllable
from simple.robots.registry import RobotRegistry
from simple.sensors import SensorCfg, StereoCameraCfg
from simple.tasks.registry import TaskRegistry

# The three colours, in a fixed order. The target (spatially randomized) cube is
# the first one; the rest are placed by this task's reset().
_COLOURS = ["red", "green", "blue"]
_TARGET_COLOUR = _COLOURS[0]

# Tabletop surface (top z). Cubes rest one half-extent (0.03) above it.
_TABLE_TOP = 0.75
_TABLE_POSITION_XY = (0.10, 0.0)   # bench centre in the canonical frame (+x = front)
_TABLE_SIZE = (0.8, 1.2, 0.1)      # x depth, y width, z thickness
_CUBE_HALF = 0.03

# Robot canonical spawn (-x side, yaw 0, facing +x). Must match robot_region.
_ROBOT_XY = (-0.64, 0.0)

# Colour pads: a fixed row across the far-reachable part of the bench, one per
# colour. Cubes are pushed forward/sideways onto the matching pad.
_PAD_X = 0.24
_PAD_Y = {"red": -0.30, "green": 0.0, "blue": 0.30}
_PAD_HALF_XY = 0.09                 # pad footprint half-extent (matches pad_*.xml)
_PAD_THICKNESS = 0.002

# Cube spawn band (near the robot, clear of the pad row in x) and placement.
_CUBE_REGION = dict(x=(-0.05, 0.10), y=(-0.35, 0.35))
_CUBE_MIN_SEP = 0.12
_CUBE_PLACE_TRIES = 200

# A cube counts as sorted when its centre is within the pad footprint (XY) and it
# is still resting on the tabletop (not knocked to the floor / stacked high).
_ON_PAD_XY_TOL = _PAD_HALF_XY
_ON_TABLE_Z = (_TABLE_TOP - 0.02, _TABLE_TOP + 0.10)

# Shader params the Isaac engine reads off every ObjectActor (obj_info.material);
# the pads/cubes are added after MaterialDR ran, so set them explicitly to the
# "fixed" branch values or the renderer crashes on the missing attribute (same
# reason as the sorting task's _FIXED_OBJECT_MATERIAL).
_FIXED_OBJECT_MATERIAL = {
    "reflection_roughness_constant": 0.5,
    "metallic_constant": 0.0,
    "specular_level": 0.0,
}


def _pad_asset(colour: str) -> ArticulatedAsset:
    """A flat colour pad as a static 0-joint articulated marker (visual only).

    The pad files live inside the ``boxes`` asset package (next to cube.obj), so
    build absolute paths from the manager's src_dir -- resolve_data_path passes
    absolute paths through unchanged.
    """
    pads_dir = os.path.join(AssetManager.get("boxes").src_dir, "pads")
    return ArticulatedAsset(
        uid=f"pad_{colour}",
        usd_path=os.path.join(pads_dir, f"pad_{colour}.usda"),
        mjcf_path=os.path.join(pads_dir, f"pad_{colour}.xml"),
        static=True,
        usd_scale=1.0,
    )


def _add_pads(layout: Layout) -> None:
    """Add the three colour pads at their fixed spots (post-DR, keys pads never
    touched by the spatial randomizer)."""
    for colour in _COLOURS:
        key = f"pad_{colour}"
        if key in layout.actors:
            continue
        layout.add_articulated_object(key, _pad_asset(colour))
        actor = layout.actors[key]
        actor.pose.position = [_PAD_X, _PAD_Y[colour], _TABLE_TOP + _PAD_THICKNESS]
        actor.pose.quaternion = [1.0, 0.0, 0.0, 0.0]


def _sample_cube_xy(occupied: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Rejection-sample a cube XY inside the spawn band, clear of `occupied`."""
    def draw() -> np.ndarray:
        return np.array([
            random.uniform(*_CUBE_REGION["x"]),
            random.uniform(*_CUBE_REGION["y"]),
        ])

    xy = draw()
    for _ in range(_CUBE_PLACE_TRIES):
        if all(np.linalg.norm(xy - pos) >= sep for pos, sep in occupied):
            return xy
        xy = draw()
    return xy  # accept the last sample rather than fail


def _add_cubes(layout: Layout, replay_cubes: list[dict] | None) -> list[dict]:
    """Place the non-target cubes (green, blue) on the bench.

    The target cube (red) is placed by the spatial randomizer; the other two are
    rejection-sampled here, clear of the target and of each other. On replay the
    recorded positions are reused so the scene matches the recording.

    Returns the spawn spec list to store in the task state_dict.
    """
    if replay_cubes is not None:
        for spec in replay_cubes:
            asset = AssetManager.get("boxes").load(spec["asset_id"])
            layout.add_object(spec["key"], asset)
            actor = layout.actors[spec["key"]]
            actor.set_material(dict(_FIXED_OBJECT_MATERIAL))
            actor.pose.position = list(spec["position"])
            actor.pose.quaternion = list(spec["quaternion"])
        return replay_cubes

    occupied: list[tuple[np.ndarray, float]] = []
    target = layout.actors.get("target")
    if target is not None:
        occupied.append((np.array(target.pose.position[:2]), _CUBE_MIN_SEP))

    spawned: list[dict] = []
    for colour in _COLOURS:
        if colour == _TARGET_COLOUR:
            continue  # the target cube is the spatially-randomized one
        asset = AssetManager.get("boxes").load(f"box_{colour}")
        xy = _sample_cube_xy(occupied)
        occupied.append((xy, _CUBE_MIN_SEP))
        yaw = random.uniform(-np.pi, np.pi)
        q = t3d.euler.euler2quat(0.0, 0.0, yaw)
        key = f"box_{colour}"
        layout.add_object(key, asset)
        actor = layout.actors[key]
        actor.set_material(dict(_FIXED_OBJECT_MATERIAL))
        actor.pose.position = [float(xy[0]), float(xy[1]), _TABLE_TOP + _CUBE_HALF]
        actor.pose.quaternion = [float(v) for v in q]
        spawned.append(
            dict(
                key=key,
                asset_id=f"box_{colour}",
                position=list(actor.pose.position),
                quaternion=list(actor.pose.quaternion),
            )
        )
    return spawned


@TaskRegistry.register("g1_fixedhand_color_box_sort_teleop")
class G1FixedHandColorBoxSortTeleop(Task):
    uid: str = "g1_fixedhand_color_box_sort_teleop"
    label: str = "G1 (fixed hand) TELEOP Colour-Box Sorting"
    description: str = (
        "A G1 robot with the fixed (rubber) hand pushes three colour cubes (red, "
        "green, blue) onto the flat pad of the matching colour on a bench."
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

    robot_cfg: dict[str, Any] = dict(uid="g1_sonic_fixedhand")

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
            pose=dict(position=[0.0, 0.0, 0.0]),
        ),
    )

    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language=LanguageDRCfg(
            instructions=[
                "Sort the cubes by colour: push each cube onto the pad of the "
                "same colour.",
            ]
        ),
        # The target is the red cube; green/blue are added in reset(). No
        # distractors (the engine names bodies by asset.label; each cube already
        # has a unique label).
        target=TargetDRCfg(asset_id=f"boxes:box_{_TARGET_COLOUR}"),
        spatial=SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(
                low=[_ROBOT_XY[0], _ROBOT_XY[1], 0.0],
                high=[_ROBOT_XY[0], _ROBOT_XY[1], 0.0],
            ),
            target_region=Box(
                low=[_CUBE_REGION["x"][0], _CUBE_REGION["y"][0]],
                high=[_CUBE_REGION["x"][1], _CUBE_REGION["y"][1]],
            ),
            target_stable_indices=[0],
            target_rotate_z=Box(low=-np.pi, high=np.pi),
            obj_surface_map={"target": "table"},
        ),
        camera=CameraDRCfg(cam_id="franka_camera"),
        scene=TabletopSceneDRCfg(
            scene_manager="warehouse",
            room_choices=["warehouse:default"],
            scene_mode="fixed",
            table_size=Box(low=list(_TABLE_SIZE), high=list(_TABLE_SIZE)),
            table_position=Box(low=list(_TABLE_POSITION_XY), high=list(_TABLE_POSITION_XY)),
            table_height=Box(low=_TABLE_TOP, high=_TABLE_TOP),
            rotation_z=Box(low=0.0, high=0.0),
            enable_table2=False,
        ),
        lighting=LightingDRCfg(
            light_mode="random",
            light_num=(2, 3),
            light_color_temperature=Box(low=3000, high=8000),
            light_intensity=Box(low=3e4, high=8e4),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 2.1),
            light_spacing=Box((1.0, 1.0), (2.5, 2.5)),
            light_position=Box((-1.1, -1.1, 1.3), (1.1, 1.1, 1.5)),
            light_eulers=Box((0, 0, -0.5 * np.pi), (0, 0, 0.5 * np.pi)),
        ),
        material=MaterialDRCfg(material_mode="fixed"),
    )

    def __init__(
        self,
        robot_uid: str = "g1_sonic_fixedhand",
        scene_uid: str | Scene = "warehouse:default",
        target_object: str | None = None,
        controller_uid: str = "pd_joint_pos",
        split: str = "train",
        render_hz: int | None = None,
        dr_level: int = 0,
        success_criteria: float = 0.9,
        *args,
        **kwargs,
    ):
        self._instruction = None
        self._target = None
        self._layout = None
        self._extra_cubes: list[dict] = []

        self.robot_cfg.update(dict(uid=robot_uid))
        self.reward = 0
        self.success_criteria = success_criteria

        self._robot = RobotRegistry.make(**self.robot_cfg, **kwargs)

        if target_object is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = target_object  # type: ignore

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
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    @property
    def observation_space(self) -> spaces.Space:
        default_obs = super().observation_space
        obs: dict[str, Any] = {
            "joint_qpos": spaces.Box(
                -np.pi, np.pi, shape=(self.robot.wholebody_dof,), dtype=np.float32
            ),  # type: ignore
        }
        if isinstance(default_obs, spaces.Dict):
            obs.update(dict(default_obs))
        return spaces.Dict(obs)

    def reset(
        self, seed: int | None = None, options: Optional[dict[str, Any]] = None
    ) -> None:
        super().reset(seed, options)
        _add_pads(self.layout)
        replay_cubes = None
        if options is not None and options.get("state_dict") is not None:
            replay_cubes = options["state_dict"].get("extra_cubes")
        self._extra_cubes = _add_cubes(self.layout, replay_cubes)
        self._target = self.layout.actors.get("target")

        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        self._instruction = lang_dr(self.metadata.get("split", "train"))

        self.reward = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({"extra_cubes": self._extra_cubes})
        return state_dict

    # ------------------------------------------------------------------ checks

    @staticmethod
    def _body_xyz(mj_model, mj_data, name: str):
        """Live world XYZ of a body, or None if the body doesn't exist."""
        try:
            bid = mj_model.body(name).id
        except Exception:
            return None
        return np.array(mj_data.xpos[bid])

    def _cube_on_pad(self, mj_model, mj_data, colour: str) -> bool:
        """True if the cube of `colour` rests on its matching-colour pad."""
        cube = self._body_xyz(mj_model, mj_data, f"box_{colour}")
        if cube is None:
            return False
        pad_xy = np.array([_PAD_X, _PAD_Y[colour]])
        on_pad = bool(np.all(np.abs(cube[:2] - pad_xy) <= _ON_PAD_XY_TOL))
        on_table = bool(_ON_TABLE_Z[0] <= cube[2] <= _ON_TABLE_Z[1])
        return on_pad and on_table

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return self.compute_reward(info, *args, **kwargs) >= self.success_criteria

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        """1/3 per cube resting on its matching-colour pad; 1.0 only when all
        three are sorted. Uses live MuJoCo body positions."""
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            self.reward = 0.0
            return self.reward

        mj_model = mujoco_env.mjModel
        mj_data = mujoco_env.mjData
        n_sorted = sum(
            int(self._cube_on_pad(mj_model, mj_data, colour)) for colour in _COLOURS
        )
        self.reward = n_sorted / len(_COLOURS)
        return self.reward

    def preload_objects(self) -> list[Actor]:
        """Preload the cube assets used by the task."""
        manager = AssetManager.get("boxes")
        return [ObjectActor(asset=manager.load(f"box_{c}")) for c in _COLOURS]
