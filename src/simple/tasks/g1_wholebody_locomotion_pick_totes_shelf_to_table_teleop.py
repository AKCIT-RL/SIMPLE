"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from simple.core.randomizer import RandomizerCfg

import numpy as np
import transforms3d as t3d
from gymnasium import spaces

from simple.assets import AssetManager
from simple.core.actor import Actor
from simple.core.layout import Layout
from simple.core.scene import Scene
from simple.core.task import Task
from simple.core.types import Pose
from simple.dr import *
from simple.dr.manager import TabletopGraspDRManager
from simple.dr.types import Box
from simple.robots.protocols import Controllable
from simple.robots.registry import RobotRegistry
from simple.sensors import SensorCfg, StereoCameraCfg
from simple.tasks.registry import TaskRegistry

# Phase 3 table calibration (docs/teleop_simple_study/toteweg_factory_scene_migration_plan.md,
# "Table calibration" -- end-cap placement, 3m clearance beyond the corridor's open -X end,
# centered on the aisle's Y midline).
_TABLE_POSITION_XY = (-7.206, 0.625)
_TABLE_HEIGHT = 0.65
_TABLE_ROTATION_Z = np.pi / 2
_TABLE_SIZE = (1.2, 0.8, 0.1)  # not yet confirmed against real reach requirements, see plan doc

# Fixed robot spawn at the aisle's Y midline (between the two shelf bands,
# y in [-0.80,-0.14] for r1 and [1.52,2.07] for l1/l3); x=0.0 sits near the
# boundary between the l-block's and r1's x ranges. Placeholder -- not yet
# confirmed against real reach/approach distance to either row, only
# verified (by real execution) to not spawn inside shelf/tote geometry and
# to keep physics stable. Set directly here rather than via `spatial` DR:
# SpatialDR unconditionally reads `layout.scene.table`, which only exists
# when a `scene` DR is also configured (confirmed by real execution, not
# just reading) -- reintroducing that coupling would reopen exactly what
# Phase 2 decided against (a SceneManager/scene DR for this corridor).
_ROBOT_SPAWN_POSITION = [0.0, 0.7, 0.0]
_ROBOT_SPAWN_QUATERNION = [0.0, 0.0, 0.0, 1.0]  # yaw 180 deg (MuJoCo wxyz convention)

# Max tilt (degrees) between a tote's local +Z axis and world +Z for it to
# still count as "upright" -- i.e. resting on the same base face it uses on
# the shelf (see TotesAsset.stable_poses in assets/totes.py, always identity
# quat, and ShelfGroupDR._yaw_quat, which only ever randomizes yaw). Eyeballed
# placeholder, not yet calibrated against a real teleop delivery -- revisit at
# Stop Point 3 of the VR guide (docs/teleop_simple_study/teleop_shelf_to_table_vr_guide.md).
_STABLE_TILT_TOLERANCE_DEG = 15.0
_WORLD_UP = np.array([0.0, 0.0, 1.0])


@TaskRegistry.register("g1_wholebody_locomotion_pick_totes_shelf_to_table_teleop")
class G1WholebodyLocomotionPickTotesShelfToTableTaskTeleop(Task):
    uid: str = "g1_wholebody_locomotion_pick_totes_shelf_to_table_teleop"
    label: str = "G1 TELEOP Pick Totes Shelf to Table"
    description: str = (
        "A task where the G1 robot must pick a tote from the estante_l1/estante_l3/estante_r1 shelves "
        "and carry it to the table at the end of the aisle."
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

    # NOTE: no `scene` DR entry here on purpose -- the corridor/table geometry
    # for this task is not an HSSD room (the only registered SceneManager);
    # `corridor0` and the table are added directly in reset() below (see the
    # migration plan doc, Phase 2 "Superseded plan" -- a factory SceneManager
    # was investigated and found incapable of producing real MuJoCo collision).
    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language=LanguageDRCfg(
            instructions=[
                "pick up a tote from the shelf and bring it to the table.",
            ]
        ),
        shelf_group=ShelfGroupDRCfg(
            asset_id="totes:toteweg",
            shelves=["l1", "l3", "r1"],
        ),
        # CameraDR.__call__ is a no-op passthrough (verified by reading) --
        # this key only exists to gate Task.reset()'s camera-registration
        # block (`if camera_dr is not None`, core/task.py). Without it, no
        # cameras are ever added to the layout regardless of sensor_cfgs,
        # confirmed by real execution (env.reset() returned only "joint_qpos",
        # missing "head_stereo_left"/"head_stereo_right" from observation_space).
        camera=CameraDRCfg(),
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
            material_mode="rand_all",
        ),
    )

    def __init__(
        self,
        robot_uid: str = "g1_sonic",
        controller_uid: str = "pd_joint_pos",
        split: str = "train",
        render_hz: int | None = None,
        dr_level: int = 0,
        success_criteria: float = 0.9,
        *args,
        **kwargs,
    ):
        self._instruction = None
        self._layout = None
        self._contact_started = False

        self.robot_cfg.update(dict(uid=robot_uid))

        self.reward = 0
        self.success_criteria = success_criteria

        self._robot = RobotRegistry.make(**self.robot_cfg, **kwargs)

        # NOTE: uses TabletopGraspDRManager (not the plain base DRManager) even
        # though this task has no tabletop-grasp roles -- the base DRManager's
        # __init__ never initializes self.randomizers (only declared as a bare
        # type-annotation, no `= {}`), so it AttributeErrors on construction.
        # Every existing task already routes around this the same way. Not
        # fixed at the DRManager level here since that's shared code touching
        # every task; flagged for a follow-up fix.
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

    def _live_target_keys(self) -> list[str]:
        """Layout keys of totes actually spawned this episode (variable count,
        see shelf_group's per-tier stochastic occupancy)."""
        return [k for k in self.layout.actors.keys() if k.startswith("target_")]

    def reset(
        self, seed: int | None = None, options: Optional[dict[str, Any]] = None
    ) -> None:
        super().reset(seed, options)
        split = self.metadata.get("split", "train")

        self._layout.actors["robot"].pose = Pose(
            position=list(_ROBOT_SPAWN_POSITION),
            quaternion=list(_ROBOT_SPAWN_QUATERNION),
        )

        # Static shelf fixture -- world-frame-baked collision/visual meshes,
        # identity pose (see migration plan doc, Phase 2).
        corridor_asset = AssetManager.get("fixtures").load("corridor0")
        self._layout.add_static_object("corridor0", corridor_asset)

        # Destination table -- Phase 3 calibration (end-cap placement beyond
        # the corridor's open -X end), built the same way TabletopSceneDR
        # builds `scene.table` (primitive:box, position.z derived from
        # table_height and half the box thickness).
        table_position = list(_TABLE_POSITION_XY) + [
            _TABLE_HEIGHT - 0.5 * _TABLE_SIZE[2]
        ]
        table_quaternion = t3d.euler.euler2quat(0, 0, _TABLE_ROTATION_Z).tolist()
        table_asset = AssetManager.create(
            "primitive:box",
            size=list(_TABLE_SIZE),
            position=table_position,
            quaternion=table_quaternion,
        )
        self._layout.add_primitive("table", table_asset)

        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        self._instruction = lang_dr(split)
        self._contact_started = False
        self.reward = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({"live_targets": self._live_target_keys()})
        return state_dict

    def _tote_is_upright(self, key: str) -> bool:
        """True if tote `key`'s local +Z axis is within
        `_STABLE_TILT_TOLERANCE_DEG` of world +Z -- i.e. resting on the same
        base face it uses on the shelf, not tipped/rotated onto a side.
        Equivalent to comparing against the asset's `stable_poses[0]`
        (always identity for toteweg, see assets/totes.py) but yaw-invariant,
        since only tilt away from upright should count as unstable (yaw is
        already unconstrained on the shelf itself, see ShelfGroupDR).
        Reads the live pose synced every step by MujocoSimulator.step()
        (engines/mujoco.py), not a stale spawn-time pose.
        """
        quat = self.layout.actors[key].pose.quaternion
        rot = t3d.quaternions.quat2mat(np.array(quat, dtype=np.float64))
        local_z_in_world = rot[:, 2]
        cos_tilt = np.dot(local_z_in_world, _WORLD_UP)
        return cos_tilt >= np.cos(np.deg2rad(_STABLE_TILT_TOLERANCE_DEG))

    def check_any_tote_on_table(self, *args, **kwargs) -> bool:
        """True if any spawned tote is resting stably on the table -- in
        contact with it AND upright (delivery goal: the robot only needs to
        bring *a* tote, not all of them, but it has to land right-side-up,
        not tipped over or balanced on an edge)."""
        return self.tote_location_counts(*args, **kwargs)["table"] > 0

    def tote_location_counts(self, *args, **kwargs) -> Dict[str, int]:
        """Single contact-scan classifying every live tote by what it's
        currently touching: 'shelf' (corridor0), 'table' (only if also
        upright, see `_tote_is_upright` -- a tote merely touching the table
        while tipped over isn't counted here, same as a tote touching
        nothing tracked), or 'ground' (fell off). Used both for the live HUD
        (X totes na estante / Y totes na mesa) and for ground-fall episode
        discarding -- one scan instead of separate `check_*` passes over the
        same contact list.
        """
        mujoco_env = kwargs.get("mujoco_env", None)
        counts = {"shelf": 0, "table": 0, "ground": 0}
        live_keys = self._live_target_keys()
        if not live_keys or mujoco_env is None:
            return counts

        # Body names follow MujocoSimulator's duplicate-label disambiguation
        # (src/simple/engines/mujoco.py): every spawned tote shares asset
        # label "toteweg", so each actual body is "toteweg_{layout_key}".
        body_name_by_key = {key: f"toteweg_{key}" for key in live_keys}
        live_body_names = set(body_name_by_key.values())

        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        tote_contacts: Dict[str, set] = {name: set() for name in live_body_names}
        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name
            for tote_name in live_body_names:
                if tote_name in body1:
                    tote_contacts[tote_name].add(body2)
                if tote_name in body2:
                    tote_contacts[tote_name].add(body1)

        for key, body_name in body_name_by_key.items():
            others = tote_contacts[body_name]
            if any("table" in o for o in others):
                if self._tote_is_upright(key):
                    counts["table"] += 1
            elif any("corridor0" in o for o in others):
                counts["shelf"] += 1
            elif any(o == "ground" for o in others):
                counts["ground"] += 1
        return counts

    def check_any_tote_on_ground(self, *args, **kwargs) -> bool:
        """True if any spawned tote has fallen off the shelf and hit the
        ground -- signals the episode should be discarded, not saved."""
        return self.tote_location_counts(*args, **kwargs)["ground"] > 0

    def check_hand_object_contact(self, *args, **kwargs) -> bool:
        """True if any live tote is still in contact with a hand -- mirrors
        `check_hand_object_contact` in the between_tables task, generalized
        to "any" tote the same way `check_any_tote_on_table` is (this task
        has no single fixed `target`). Used to require an actual release
        before delivery counts, not just the tote resting against the table
        while still gripped/pressed by the hand."""
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False

        live_body_names = {f"toteweg_{key}" for key in self._live_target_keys()}
        if not live_body_names:
            return False

        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name

            body1_is_tote = any(name in body1 for name in live_body_names)
            body2_is_tote = any(name in body2 for name in live_body_names)
            if (body1_is_tote and "hand" in body2) or (body2_is_tote and "hand" in body1):
                return True
        return False

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info, *args, **kwargs)
        return reward >= self.success_criteria

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        is_tote_on_table = self.check_any_tote_on_table(*args, **kwargs)
        is_tote_contacted_by_hand = self.check_hand_object_contact(*args, **kwargs)

        if is_tote_on_table and not is_tote_contacted_by_hand:
            self.reward += 0.02
        else:
            self.reward = 0.0
        return self.reward

    def preload_objects(self) -> list[Actor]:
        """Preloads all assets required by the task."""
        from simple.core.actor import ObjectActor

        asset_manager = AssetManager.get("totes")
        return [ObjectActor(asset=asset) for asset in asset_manager]
