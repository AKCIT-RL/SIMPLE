"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

G1 (fixed/rubber hand) loco-manip COLOUR-TOTE task.

Four colour-coded totes sit on the source assembly bench (table1); one is the
target (named by colour in the prompt) and the other three are distractors. The
robot must move the TARGET-colour tote onto the dispatch workbench (table2,
across the room), then release it. Bench disposition matches the industrial
screwdriver-to-tote task. It is a language-conditioned selection task: the colour
in the prompt is the only thing that says which tote to move.

Control: the standard ``decoupled_wbc`` locomotion stack + the fixed-hand robot
(robots/g1_sonic_fixedhand.py). The rubber hand has no fingers, so it cannot
grasp a handle -- the intended skill is a BIMANUAL HUG (press the tote between
the two hands/forearms) and carry it the short distance to the adjacent table.
The task CODE is agnostic to HOW the tote is carried: success is simply the
target tote coming to rest on table2, so pushing/sliding also counts.

Colours are a hardcoded dict (``_TOTE_COLORS``). Every tote reuses the bin_b04
mesh/USD; the colour comes from ``asset.rgba`` (MuJoCo geom tint / Isaac OmniPBR
bind), and each tote gets a unique label so several can coexist in one MuJoCo
scene. The target tote is the spatially-randomized one (tinted to the target
colour in reset); the three distractors are placed here as labeled copies.

Success (episode completion): the target tote rests on table2 (a live tote<->
table2 contact AND the tote centre within the table2 footprint). Checks use live
MuJoCo state, not the initial layout poses.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np
import transforms3d as t3d
from gymnasium import spaces

from simple.assets import AssetManager
from simple.core.actor import Actor, ObjectActor
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

# Hardcoded tote colour dict (uid suffix -> RGBA). The colour name is the
# language cue in the prompt; there are four so the target colour names exactly
# one tote (target + 3 distractors = 4 totes, 4 distinct colours).
_TOTE_COLORS: dict[str, list[float]] = {
    "red": [0.90, 0.02, 0.02, 1.0],
    "green": [0.02, 0.70, 0.05, 1.0],
    "blue": [0.02, 0.08, 0.90, 1.0],
    "yellow": [0.90, 0.80, 0.02, 1.0],
}
_COLORS = list(_TOTE_COLORS)

# Bench disposition matches the industrial screwdriver-to-tote task: a wide
# assembly bench in front of the robot (table1) and a dispatch workbench across
# the room (table2, reached by turning around and walking). Same size/pos/height.
_TABLE_TOP = 0.75
_TABLE1_POSITION_XY = (0.0, 0.25)
_TABLE1_SIZE = (0.8, 2.0, 0.1)
_TABLE2_POSITION_XY = (-2.5, 0.25)
_TABLE2_SIZE = (0.8, 2.0, 0.1)

_ROBOT_XY = (-0.64, 0.0)

# Four tote slots along the near (reachable) half of table1, spread in y so the
# totes never overlap. The target tote lands in slot 0 (via the spatial
# randomizer's target_region); the three distractors fill slots 1..3.
_TOTE_X = -0.10
_TOTE_SLOTS_Y = [-0.35, 0.05, 0.45, 0.85]
_TOTE_XY_JITTER = 0.04

# A tote counts as delivered when it touches table2 AND its centre is within the
# table2 footprint (minus a small margin so an overhanging tote does not count).
_TABLE2_XY_MARGIN = 0.10
_TOTE_CONTACT_MARGIN = 0.005

# Shader params the Isaac engine reads off every ObjectActor (see the sorting
# task); totes are added after MaterialDR ran, so set them explicitly.
_FIXED_OBJECT_MATERIAL = {
    "reflection_roughness_constant": 0.5,
    "metallic_constant": 0.0,
    "specular_level": 0.0,
}


def _load_colored_tote(color: str, label: str):
    """A bin_b04 tote copy tinted to `color` with a unique uid/label.

    load() builds a fresh asset per call, so mutating rgba/label/uid here never
    leaks into other instances. Unique labels are required because the MuJoCo
    engine names bodies by asset.label (duplicates fail to compile).
    """
    asset = AssetManager.get("totes").load("bin_b04")
    asset.rgba = list(_TOTE_COLORS[color])
    asset.label = label
    asset.uid = label
    asset.contact_margin = _TOTE_CONTACT_MARGIN
    return asset


def _table1_top(layout: Layout) -> float | None:
    table = getattr(layout.scene, "table", None)
    if table is None:
        return None
    return table.pose.position[2] + 0.5 * table.size[2]


def _spawn_distractor_totes(
    layout: Layout, distractor_colors: list[str], replay: list[dict] | None
) -> list[dict]:
    """Place the (non-target) distractor totes on table1 as labeled copies.

    On replay the recorded colours/positions are reused so the scene matches the
    recording. Returns the spawn spec list to store in the task state_dict.
    """
    top = _table1_top(layout)
    if top is None:
        return []

    if replay is not None:
        for spec in replay:
            asset = _load_colored_tote(spec["color"], spec["label"])
            layout.add_object(spec["key"], asset)
            actor = layout.actors[spec["key"]]
            actor.set_material(dict(_FIXED_OBJECT_MATERIAL))
            actor.pose.position = list(spec["position"])
            actor.pose.quaternion = list(spec["quaternion"])
        return replay

    # Fixed, well-separated slots (slot 0 is the spatially-randomized target, so
    # the three distractors take slots 1..3). Deterministic placement guarantees
    # the four totes never overlap; small jitter keeps some variety.
    spawned: list[dict] = []
    for i, color in enumerate(distractor_colors):
        label = f"tote_{color}"
        asset = _load_colored_tote(color, label)
        sx = _TOTE_X + random.uniform(-_TOTE_XY_JITTER, _TOTE_XY_JITTER)
        sy = _TOTE_SLOTS_Y[i + 1] + random.uniform(-_TOTE_XY_JITTER, _TOTE_XY_JITTER)
        yaw = random.uniform(-np.pi, np.pi)
        q = t3d.euler.euler2quat(0.0, 0.0, yaw)
        stable_z = float(asset.stable_poses[0][2])
        layout.add_object(label, asset)
        actor = layout.actors[label]
        actor.set_material(dict(_FIXED_OBJECT_MATERIAL))
        actor.pose.position = [float(sx), float(sy), top + stable_z]
        actor.pose.quaternion = [float(v) for v in q]
        spawned.append(
            dict(
                key=label,
                label=label,
                color=color,
                position=list(actor.pose.position),
                quaternion=list(actor.pose.quaternion),
            )
        )
    return spawned


@TaskRegistry.register("g1_fixedhand_tote_to_adjacent_table_teleop")
class G1FixedHandToteToAdjacentTableTeleop(Task):
    uid: str = "g1_fixedhand_tote_to_adjacent_table_teleop"
    label: str = "G1 (fixed hand) TELEOP Colour-Tote to Adjacent Table"
    description: str = (
        "A G1 robot with the fixed (rubber) hand moves the target-colour tote "
        "(named in the prompt) from the source bench to the adjacent table, "
        "among three distractor totes of other colours."
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
                "Pick up the {color} tote and place it on the other table.",
            ]
        ),
        # The target is the spatially-randomized tote (tinted to the target
        # colour in reset). Distractor totes are added in reset() as labeled
        # copies -- no DistractorDRCfg, which cannot spawn repeated base assets
        # (the engine names bodies by asset.label, so instances need unique
        # labels).
        target=TargetDRCfg(asset_id="totes:bin_b04"),
        spatial=SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(
                low=[_ROBOT_XY[0], _ROBOT_XY[1], 0.0],
                high=[_ROBOT_XY[0], _ROBOT_XY[1], 0.0],
            ),
            target_region=Box(
                low=[_TOTE_X - _TOTE_XY_JITTER, _TOTE_SLOTS_Y[0] - 0.05],
                high=[_TOTE_X + _TOTE_XY_JITTER, _TOTE_SLOTS_Y[0] + 0.05],
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
            table_size=Box(low=list(_TABLE1_SIZE), high=list(_TABLE1_SIZE)),
            table_position=Box(low=list(_TABLE1_POSITION_XY), high=list(_TABLE1_POSITION_XY)),
            table_height=Box(low=_TABLE_TOP, high=_TABLE_TOP),
            rotation_z=Box(low=0.0, high=0.0),
            enable_table2=True,
            table2_size=Box(low=list(_TABLE2_SIZE), high=list(_TABLE2_SIZE)),
            table2_position=Box(low=list(_TABLE2_POSITION_XY), high=list(_TABLE2_POSITION_XY)),
            table2_height=Box(low=_TABLE_TOP, high=_TABLE_TOP),
            table2_rotation_z=Box(low=0.0, high=0.0),
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
        self._target_color: str = _COLORS[0]
        self._distractors: list[dict] = []

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
    def target_color(self) -> str:
        """The colour of the tote to deliver this episode (drives the prompt).

        Read by the teleop CLI to build the VR HUD / for balanced collection.
        """
        return self._target_color

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
        self._target = self.layout.actors.get("target")

        # Target colour: replay reuses the recorded value; fresh episodes sample.
        replay_color = None
        replay_distractors = None
        if options is not None and options.get("state_dict") is not None:
            replay_color = options["state_dict"].get("target_color")
            replay_distractors = options["state_dict"].get("distractors")
        self._target_color = replay_color if replay_color is not None else random.choice(_COLORS)

        # Tint the (spatially-randomized) target tote to the target colour. Set
        # before the engine builds the MuJoCo/Isaac scene (this runs in
        # task.reset, before mujoco.update_layout), so the tint is applied.
        if self._target is not None:
            self._target.asset.rgba = list(_TOTE_COLORS[self._target_color])
            self._target.asset.contact_margin = _TOTE_CONTACT_MARGIN
            self._target.set_material(dict(_FIXED_OBJECT_MATERIAL))

        distractor_colors = [c for c in _COLORS if c != self._target_color]
        self._distractors = _spawn_distractor_totes(
            self.layout, distractor_colors, replay_distractors
        )

        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        self._instruction = lang_dr(self.metadata.get("split", "train")).format(
            color=self._target_color
        )

        self.reward = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update(
            {
                "target_color": self._target_color,
                "distractors": self._distractors,
            }
        )
        return state_dict

    # ------------------------------------------------------------------ checks

    def _target_on_table2(self, mujoco_env) -> bool:
        """True if the target tote rests on table2 (live contact + XY footprint)."""
        mj_model = mujoco_env.mjModel
        mj_data = mujoco_env.mjData
        target_name = str(self.target.asset.label)

        # tote centre within the table2 footprint (minus a margin)
        try:
            tote_xy = np.array(mj_data.xpos[mj_model.body(target_name).id][:2])
        except Exception:
            return False
        half = 0.5 * np.array(_TABLE2_SIZE[:2]) - _TABLE2_XY_MARGIN
        within = bool(np.all(np.abs(tote_xy - np.array(_TABLE2_POSITION_XY)) <= half))
        if not within:
            return False

        # live tote <-> table2 contact
        for i_contact in range(mj_data.ncon):
            contact = mj_data.contact[i_contact]
            b1 = mj_model.body(mj_model.geom(contact.geom1).bodyid).name
            b2 = mj_model.body(mj_model.geom(contact.geom2).bodyid).name
            if (target_name in b1 and "table2" in b2) or (
                target_name in b2 and "table2" in b1
            ):
                return True
        return False

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return self.compute_reward(info, *args, **kwargs) >= self.success_criteria

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        """1.0 when the target-colour tote rests on the adjacent table (table2),
        else 0.0. Uses live MuJoCo state."""
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            self.reward = 0.0
            return self.reward
        self.reward = 1.0 if self._target_on_table2(mujoco_env) else 0.0
        return self.reward

    def preload_objects(self) -> list[Actor]:
        """Preload the tote asset used by the task."""
        manager = AssetManager.get("totes")
        return [ObjectActor(asset=manager.load("bin_b04"))]
