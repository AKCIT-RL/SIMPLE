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
import transforms3d as t3d
from simple.assets import AssetManager

_LIFT_HEIGHT = 0.02

# Wheel body pos z (0.0345) - wheel radius (0.098), from data/robots/miss/miss.xml's
# front/rear_{left,right}_wheel_link geoms -- confirmed by direct rendering (the wheels were
# visibly sunk into the floor at robot z=0). Placing the robot's own origin (chassis base) at this
# height makes the wheel bottoms land exactly on world z=0, i.e. the Jackal base actually touches
# the floor, instead of resting ~6.35cm below it.
_ROBOT_Z_OFFSET = 0.098 - 0.0345

# hssd:scene1's own table2_size[:2] (SceneManager.get("hssd").load("hssd:scene1").conf) -- this
# is the REAL, functional desk's footprint (see the class docstring's "table vs table2" finding).
# Half-depth used for the chassis-clearance calculation below. Re-check this against
# SceneManager.get("hssd").load(scene_uid).conf["table2_size"] if `scene_uid` is ever changed
# away from the default -- this is not necessarily the same for every HSSD scene.
_TABLE2_HALF_DEPTH_X = 2.0 / 2


@TaskRegistry.register("miss_tabletop_grasp_mp")
class MissTabletopGraspTaskMP(Task):
    """Minimal validation task for the Miss platform: spawn next to a desk with a banana on it.

    Deliberately minimal (no distractors, a single fixed target) -- this exists to validate the
    whole Miss integration (arm + gripper + gimbal camera) end-to-end, mirroring how
    `viperx_tabletop_grasp_mp.py`/`widowx_ai_tabletop_grasp_mp.py` served the same role for their
    own robots, not to be a tuned benchmark task. See "Known limitations" in
    docs/source/miss/miss_integration.md.

    **Desk placement knobs** -- explicit, adjustable constructor parameters (not DR-sampled
    ranges; each is threaded in as a degenerate `Box(low=x, high=x)`, the same pattern every DR
    config in this codebase uses to express a "fixed" value):
    - `table_distance` (m): how far the desk center is from the robot's own origin.
    - `table_angle` (rad): the direction from the robot to the desk center (0 = the robot's local
      +x axis). Together with `table_distance` this sets the desk's (x, y) position.
    - `table_height` (m): the desk's height off the floor.
    - `robot_yaw` (rad): the robot's own heading relative to the desk. This is how "orientation of
      the robot to the table" is actually implemented -- `TabletopSceneDRCfg`'s `rotation_z`
      field has **no effect** when `scene_manager="hssd"` (confirmed by reading
      `TabletopSceneDR.__call__` in `src/simple/dr/scene.py`: the hssd branch hardcodes
      `rotation_z = 0.0` and never reads `self.cfg.rotation_z` at all), so rotating the desk
      itself isn't possible in this scene mode. Only the *relative* angle between robot and desk
      matters for a single fixed robot + single table scene, so rotating the robot instead
      (`SpatialDRCfg.robot_orientation_region`) achieves the same effect and is fully supported.
    - `object_distance` (m) / `object_angle` (rad): where the banana spawns, using the same
      "direction from robot origin" convention as `table_distance`/`table_angle` but **entirely
      independent of it**. `table2` (see below) is `2.0m` deep, so spawning the banana at its own
      center (`table_distance` away) would put it far past anything the arm has reached in any FK/
      IK check performed on Miss so far (<1m) -- these two knobs let the banana sit much closer to
      the robot than the table's own center, while still landing somewhere on `table2`'s actual
      footprint (checked at construction time; a combination that would spawn the banana off the
      desk prints a warning rather than failing silently).

    `table_height=0.70` and `robot_yaw=0.0` are reasonable real-world starting points, not tuned
    for reachability.

    **CRITICAL FINDING, found by comparing the DR-level pose against the actually-compiled MuJoCo
    body (not just trusting `task.layout.scene.table.pose`): the desk this task used through two
    earlier rounds of `table_distance` correction was never actually elevated in world
    coordinates at all.** `MujocoEngine._build_primitive` (`src/simple/engines/mujoco.py`)
    hardcodes, for any actor named exactly `"table"`: `table_position[2] = -0.5 * actor.size[2]`
    -- **unconditionally overriding whatever Z the DR computed**, including any `table_height`
    offset. This is not a bug in isolation -- it is the deliberate mechanism every other tabletop
    task in this codebase (Franka/Aloha/ViperX/WidowX AI) relies on: those robots are bolted
    directly to the tabletop, so *their own* `robot_region`'s 2-element `[x, y]` form shifts the
    *robot* down by `-table_height` (see the `robot_region` note below) so that, relative to the
    robot's own shifted-down origin, the always-floor-level "table" appears to be at the right
    height. **Miss can't use that trick** -- its base genuinely needs to sit at world Z=0 (wheels
    on the floor), which is exactly why the 3-element `robot_region` form was used instead (see
    below) to opt out of the `-table_height` shift. The unintended consequence: with the robot no
    longer shifted down, and the table's engine-level Z hardcoded to floor level regardless of any
    DR value, **both the robot and the "table" actor sat at the same world height** -- confirmed
    directly (`d.xpos[table_body_id]` was `[1.4, 0.0, -0.05]` even when `table_height=0.9` was
    requested and `task.layout.scene.table.pose.position` correctly said `z=0.85`) -- exactly the
    "robot and desk spawning on the same plane" the user reported, and *not* fixed by either of
    the two earlier `table_distance` correction rounds (both only ever address horizontal
    clearance, which remains independently necessary -- see below -- but neither could have fixed
    a vertical placement bug neither of us had found yet).

    **The fix**: `_build_primitive`'s hardcoded Z-override only applies when `table_name ==
    "table"` -- its own comment says "allow personalization of other tables", pointing at the
    parallel `table2` mechanism (`TabletopSceneDRCfg.enable_table2`/`table2_position`/
    `table2_height`/`table2_rotation_z`, `src/simple/dr/scene.py`), which places its body using
    the DR-computed pose directly, with **no engine-level override** -- confirmed directly
    (`table2`'s compiled `xpos` matched `task.layout.scene.table2.pose.position` exactly,
    `z=0.65` for `table_height=0.70`). This task now uses `table2` as the real, functional desk:
    `enable_table2=True`, `table2_position`/`table2_height` driven by the same
    `table_distance`/`table_angle`/`table_height` knobs, and `SpatialDRCfg.obj_surface_map =
    {"target": "table2"}` so the banana spawns on `table2`'s real elevated surface instead of the
    (now-vestigial) floor-level `"table"`'s. The old `"table"` actor is still created (the hssd
    scene-building code path creates it unconditionally, no way to opt out) but is now understood
    to be inert set-dressing at floor level, not the functional desk -- its `table_position` is
    still set to the same `(table_x, table_y)` as `table2` purely so its physical footprint stays
    directly *underneath* `table2` rather than sitting exposed somewhere else in the room; there is
    no Z overlap between the two (`"table"` spans roughly `z ∈ [-0.1, 0]`, `table2`'s underside is
    at `table_height - 0.05`, well above).

    **This also changes the desk's real footprint**, and therefore the chassis-clearance math:
    `table2`'s footprint comes from `scene.conf["table2_size"][:2]` (hardcoded per-HSSD-scene, NOT
    the same field as `"table"`'s own `table_size`) -- for `hssd:scene1` this is `[2.0, 0.7]`, not
    `"table"`'s `[1.27, 0.7112]` used in the earlier (still horizontally-necessary, but now
    superseded) clearance calculation. Redone with the correct footprint:
    `table_distance >= 2.0/2 (half-depth) + 0.21 (chassis half-length) + 0.5 (desired clearance) =
    1.71`, rounded up to `1.75` (~0.55m of actual clearance) -- see `_TABLE2_HALF_DEPTH_X`.

    **Still unverified**: whether `hssd:scene1`'s own room geometry (walls, doorways) is even
    large enough to contain the desk at this distance without clipping through a wall -- check
    visually with `scripts/test_miss_task_scene.py` before trusting this further. **Also still
    open**: `gimbal_cam`'s field-of-view framing (established for the old, incorrect `"table"`
    setup, at completely different distances/heights) needs re-verification from scratch against
    this corrected scene, not assumed to still apply.

    **`robot_region` is set as an explicit 3-element `[x, y, z]` `Box`, not the more common
    2-element `[x, y]` form.** `SpatialDR` (`src/simple/dr/spatial.py`) special-cases a 2-element
    `robot_region` by setting the robot's world Z to `-table_height` -- correct for a robot bolted
    directly onto the tabletop (Franka/Aloha/ViperX/WidowX AI all rest at the table's own height),
    but wrong for Miss: it's a wheeled platform whose own base touches the floor regardless of the
    desk's height. Using `[x, y, _ROBOT_Z_OFFSET]` (3 elements) both bypasses that special-case and
    corrects for the robot's own root frame not being at wheel-bottom height -- see
    `_ROBOT_Z_OFFSET`'s own comment (computed from `miss.xml`'s wheel geometry, confirmed by
    direct rendering: at Z=0 the wheels were visibly sunk ~6.35cm into the floor).
    """

    uid: str = "miss_tabletop_grasp_mp"
    label: str = "Miss Tabletop Grasp Task"
    description: str = "A minimal task where the Miss platform must grasp a banana on a desk."

    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "dr_level": 0,
        "version": 1.0,
    }

    robot_cfg: dict[str, Any] = dict(
        uid="miss",
    )

    sensor_cfgs: dict[str, SensorCfg] = dict(
        # The gimbal-mounted D455 is Miss's defining sensor (it has no wrist camera, unlike
        # Aloha/ViperX) -- mount="native" needs no engine changes, the same mechanism already
        # proven for WidowX AI's/Aloha's own wrist cameras (MujocoEngine._build_camera just
        # asserts the named <camera> element already exists in the robot's own MJCF and lets
        # MuJoCo's kinematics update its world pose every step as head_pan/head_tilt move).
        # gimbal_cam's own orientation is still an open TODO (see miss.xml/miss_integration.md) --
        # this task is one of the first places that can actually be used to verify/fix it.
        gimbal_cam = CameraCfg(
            uid="Realsense_D455",
            mount="native",
            width=480,
            height=270,
            focal_length=1.93,
            fov=np.deg2rad(87),
            near=0.1,
            far=5,
            pose=dict(position=[0.0, 0.0, 0.0], quaternion=[1.0, 0.0, 0.0, 0.0]),  # unused by mount="native"
        ),
        # Fixed overview camera, same eye_on_base pattern as every other tabletop task -- but with
        # very different distance/polar/azimuth values than any of them. CameraEntity's spherical
        # pose (src/simple/core/actor.py) always looks at world-frame [0, 0, 0], which for a
        # compact tabletop arm (Franka/Aloha/ViperX/WidowX AI) is right at the robot's own
        # workspace height, so any reasonable distance/polar/azimuth frames the arm+table well.
        # Miss's origin is its *floor-level chassis base* -- confirmed by rendering every
        # combination tried at the other tasks' typical distance (~1.5m)/polar (~55-65deg): the
        # arm+desk (up around z=0.75-1.3m) never appeared, or only clipped at the frame edge. This
        # configuration (larger distance, larger/flatter polar angle) was found by directly
        # rendering several combinations until both the arm+gripper and the desk were visible in
        # the same frame -- not derived analytically, and not claimed optimal.
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
                distance=2.5,
                polar=np.deg2rad(75),
                azimuth=np.deg2rad(45),
            )
        ),
    )

    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language = LanguageDRCfg(
            instructions = [
                "Pick up {}.",
            ]
        ),

        # Fixed target, no distractors -- deliberately minimal, see class docstring.
        target = TargetDRCfg(
            asset_id="graspnet1b:5"  # banana
        ),

        distractors = DistractorDRCfg(
            res_id="graspnet1b",
            number_of_distractors=0,
            allow_duplicates=False,
            exclude=["5"]
        ),

        # Placeholder region -- overwritten in __init__ from the table_distance/table_angle/
        # table_height/robot_yaw knobs (see class docstring). NOT independently tuned yet.
        # obj_surface_map routes the target's surface-height lookup to "table2" -- the real,
        # correctly-elevated desk -- instead of the default "table" (floor-level, see docstring).
        spatial = SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[0.0, 0.0, _ROBOT_Z_OFFSET], high=[0.0, 0.0, _ROBOT_Z_OFFSET]),
            target_region=Box(low=[0.6, -0.1], high=[0.6, 0.1]),
            distractors_region=Box(low=[0.4, -0.3], high=[0.8, 0.3]),
            obj_surface_map={"target": "table2"},
        ),

        camera = CameraDRCfg(
            cam_id="miss_camera",
        ),

        # "table" is created unconditionally by the hssd scene-building code path but is NOT the
        # functional desk for this task -- see the class docstring's "table vs table2" finding.
        # table_height here is effectively vestigial (MujocoEngine._build_primitive ignores it for
        # the "table" actor specifically) but left at a plausible value rather than 0, in case
        # anything else ever reads it. enable_table2 turns on the real, correctly-elevated desk;
        # table2_position/table2_height are set from the table_distance/table_angle/table_height
        # knobs in __init__, same as table_position/table_height are for the vestigial "table".
        scene = TabletopSceneDRCfg(
            scene_mode="fixed",
            table_height=Box(low=0.75, high=0.75),
            scene_manager="hssd",
            enable_table2=True,
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
        robot_uid: str = "miss",
        target_object: str | None = "graspnet1b:5",
        scene_uid: str | None = None,
        split: str = "train",
        render_hz: int | None = None,
        dr_level: int = 0,
        physics_dt: float = 0.002,
        table_distance: float = 1.55,
        table_angle: float = 0.0,
        table_height: float = 0.80,
        robot_yaw: float = 0.0,
        object_distance: float = 0.65,
        object_angle: float = 0.0,
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

        # Desk placement knobs -- see class docstring for the full explanation of why these four
        # parameters (and not e.g. a table rotation_z) are how "distance/pose/orientation of the
        # robot to the table" is exposed.
        assert isinstance(self.dr_cfgs["scene"], TabletopSceneDRCfg)
        table_x = table_distance * math.cos(table_angle)
        table_y = table_distance * math.sin(table_angle)
        # "table" (vestigial, floor-level regardless of table_height -- see class docstring) kept
        # centered under table2 so its physical footprint never sits exposed elsewhere in the room.
        self.dr_cfgs["scene"].table_position = Box(low=[table_x, table_y], high=[table_x, table_y])
        self.dr_cfgs["scene"].table_height = Box(low=table_height, high=table_height)
        # "table2" is the REAL, correctly-elevated desk this task actually uses -- see class
        # docstring's "table vs table2" finding. Driven by the exact same knobs as "table" above.
        self.dr_cfgs["scene"].table2_position = Box(low=[table_x, table_y], high=[table_x, table_y])
        self.dr_cfgs["scene"].table2_height = Box(low=table_height, high=table_height)

        # The banana's own spawn position is a SEPARATE knob from table_distance/table_angle --
        # table2 is 2.0m deep (see _TABLE2_HALF_DEPTH_X), so spawning the banana at table2's own
        # center (the original behavior) would put it ~1.75m from the robot, well past anything
        # the arm has ever reached in FK/IK checks so far (<1m). object_distance/object_angle use
        # the same "direction from robot origin" convention as table_distance/table_angle, letting
        # the banana be placed much closer to the robot than the table's own center.
        assert isinstance(self.dr_cfgs["spatial"], SpatialDRCfg)
        object_x = object_distance * math.cos(object_angle)
        object_y = object_distance * math.sin(object_angle)
        # table2 is never rotated in this task (table2_rotation_z is left at its own default,
        # 0.0), so its depth/width axes align with world X/Y -- checking the object stays within
        # table2's real footprint (not just "somewhere near the table") catches a genuinely
        # possible mistake (an object_distance/object_angle combination that floats past the
        # table's edge) rather than silently spawning the banana off the desk.
        half_depth, half_width = _TABLE2_HALF_DEPTH_X, 0.7 / 2
        if abs(object_x - table_x) > half_depth or abs(object_y - table_y) > half_width:
            print(
                f"WARNING: object_distance={object_distance}/object_angle={object_angle} "
                f"(-> [{object_x:.3f}, {object_y:.3f}]) falls outside table2's own footprint "
                f"(center [{table_x:.3f}, {table_y:.3f}], half-extents ±{half_depth}/±{half_width}) "
                f"-- the banana will spawn off the desk."
            )
        self.dr_cfgs["spatial"].target_region = Box(
            low=[object_x - 0.05, object_y - 0.05], high=[object_x + 0.05, object_y + 0.05]
        )

        assert isinstance(self.dr_cfgs["spatial"], SpatialDRCfg)
        robot_quat = t3d.euler.euler2quat(0.0, 0.0, robot_yaw).tolist()  # wxyz
        self.dr_cfgs["spatial"].robot_orientation_region = Box(low=robot_quat, high=robot_quat)

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
