"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

G1 industrial multi-piece SORTING task (v2), adapted from the Isaac Sim scene
`ref_map/IH_basic.usda`. The robot sorts parts on the assembly bench into two
color-coded totes and dispatches each tote to its shelf:

    screws       -> RED tote  -> LEFT shelf
    screwdrivers -> BLUE tote -> RIGHT shelf

Scene (all furniture with REAL mesh collision — Plan A, articulated 0-joint
attach path; see scripts/industrial/ and INDUSTRIAL_IH_TASK_PROGRESS.md):

  * TableTrolley (assembly bench) at the origin of the layout, tabletop 0.854 m.
  * TWO MobileShelvingCart instances as the LEFT and RIGHT shelves. The
    GravityShelfBinOrganizer from v1 was REMOVED for performance: its visual
    mesh alone is 213k verts / 392k faces (vs 23k verts per cart), and it needed
    48 collision hulls. The carts reuse the same meshes on disk via renamed
    attach XMLs (scripts/industrial/make_shelf_variants.py) because the engine
    attaches furniture without a name prefix and duplicate names don't compile.
  * Totes are `totes:bin_b04_red` / `totes:bin_b04_blue` — color variants that
    reuse the bin_b04 meshes but carry distinct uid/label (so both can coexist
    in one MuJoCo scene) and an RGBA tint the MuJoCo engine applies to their
    geoms. NOTE: the tint is MuJoCo-side; the Isaac Sim replay renders the
    bin_b04 USD material (color override there belongs to the render stage).

Robot: spawns at the CANONICAL pose (-x side of the bench, yaw 0, facing +x).
The WBC teleop stack (leg policy + hand retargeting) is calibrated for this
heading — spawning the robot yawed 180 broke VR responsiveness. We still want the
OPERATOR composition (robot at the +x side facing the bench), so the scene is
authored in that operator frame and rigidly mapped onto the canonical robot: a
180deg planar rotation about the midpoint of the operator/canonical robot
positions (p -> S - p, yaw -> yaw + pi; see _op_to_canon_*). This keeps the
robot<->furniture relative pose IDENTICAL to the operator arrangement while the
robot physically stays at the teleop-safe spawn. In the canonical frame the
robot's LEFT is +y and its RIGHT is -y: red tote / left shelf on +y, blue tote /
right shelf on -y.

Part-count DR (v2.1): each episode spawns 1..4 screws and 1..4 screwdrivers on
the bench (_spawn_parts). The spatial-DR target is always screw #1; the rest are
labeled copies of graspnet assets (unique labels — the engine names bodies by
asset.label). The spawn list is stored in state_dict["extra_parts"] so replay
recreates the exact same bodies.

Success (and episode completion): reward 0.25 per condition —
  [>=1 screw in red tote] + [red tote resting on LEFT shelf] +
  [>=1 screwdriver in blue tote] + [blue tote resting on RIGHT shelf]
so with success_criteria = 0.9 the episode only completes when ALL four hold.
Checks use live MuJoCo state (body xpos + geom<->geom contacts), not the layout's
initial poses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from simple.core.randomizer import RandomizerCfg

from typing import Any

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

# Tabletop height pinned to the TableTrolley tabletop so the primitive `table`
# surface (where spatial DR drops the parts) coincides with the trolley mesh top.
_BENCH_TOP = 0.854

# How close (XY, meters) a part must be to a tote's center — combined with a
# live geom contact — to count as "inside" it.
_IN_TOTE_XY_TOL = 0.12

_SHELF_LEFT = "MobileShelvingCart_C05_left"
_SHELF_RIGHT = "MobileShelvingCart_C05_right"
_TOTE_RED = "bin_b04_red"
_TOTE_BLUE = "bin_b04_blue"

# --- operator layout -> canonical spawn --------------------------------------
# The robot MUST spawn at the CANONICAL teleop-safe pose (-x side, yaw 0, facing
# +x); rotating the robot to the operator side (yaw pi) broke VR responsiveness.
# So the scene is authored in the OPERATOR frame — robot standing at the +x side
# facing the bench, which is the composition we want — and every element is
# rigidly mapped onto the canonical robot so the robot<->furniture relative pose
# is IDENTICAL to the operator arrangement. The transform that carries the
# operator robot pose onto the canonical one is a 180deg planar rotation about
# the midpoint of the two robot positions:  p -> S - p  (S = op_xy + canon_xy),
# yaw -> yaw + pi. (validate_sorting_task.py stage 5 proves the relative poses
# match to numerical tolerance.)
_OP_ROBOT_XY = np.array([0.84, 0.25])       # operator robot stands here (yaw pi)
_CANON_ROBOT_XY = np.array([-0.64, 0.25])   # actual spawn = robot_region middle (yaw 0)
_S = _OP_ROBOT_XY + _CANON_ROBOT_XY


def _op_to_canon_xy(x: float, y: float) -> tuple[float, float]:
    """Map a point from the operator frame onto the canonical frame (p -> S - p)."""
    return (float(_S[0] - x), float(_S[1] - y))


def _op_to_canon_yaw(yaw: float) -> float:
    """Map a yaw from the operator frame onto the canonical frame: wrap(yaw + pi)."""
    return float((yaw + 2 * np.pi) % (2 * np.pi) - np.pi)


def _op_to_canon_region(low, high):
    """A 180deg map p->S-p sends the AABB [low,high] to [S-high, S-low]."""
    lo = (float(_S[0] - high[0]), float(_S[1] - high[1]))
    hi = (float(_S[0] - low[0]), float(_S[1] - low[1]))
    return lo, hi


# Bench position in the operator frame (also the `table` primitive position
# there). The bench maps into the canonical frame and its trolley mesh is yawed
# pi so the operator face points at the robot.
_BENCH_OP_XY = (0.0, 0.25)
_BENCH_CANON_XY = _op_to_canon_xy(*_BENCH_OP_XY)

# Furniture authored in OPERATOR world coords; op_xy/op_yaw are mapped to the
# canonical frame at placement time (_add_furniture). Shelves keep their IH spots
# relative to the bench; the left shelf (red tote destination) is on the
# operator's left (-y), the right shelf (blue) on the operator's right (+y).
_FURNITURE: list[dict[str, Any]] = [
    dict(
        key="furniture_bench",
        name="TableTrolley_B02_01",
        folder="TableTrolley_B02_01",
        mjcf="TableTrolley_B02_01_attach.xml",
        op_xy=_BENCH_OP_XY,
        op_yaw=0.0,
    ),
    dict(
        key="furniture_shelf_left",  # operator's left = -y
        name=_SHELF_LEFT,
        folder="MobileShelvingCart_C05_01",
        mjcf=f"{_SHELF_LEFT}_attach.xml",
        op_xy=(0.852, -1.799),
        op_yaw=0.942,
    ),
    dict(
        key="furniture_shelf_right",  # operator's right = +y
        name=_SHELF_RIGHT,
        folder="MobileShelvingCart_C05_01",
        mjcf=f"{_SHELF_RIGHT}_attach.xml",
        op_xy=(0.397, 2.087),
        op_yaw=-0.942,
    ),
]

# Totes authored in OPERATOR world coords: red on the operator's left (-y, → left
# shelf), blue on the operator's right (+y, → right shelf).
_TOTES: list[dict[str, Any]] = [
    dict(key="tote_red", asset_id=f"totes:{_TOTE_RED}", op_xy=(0.15, -0.55)),
    dict(key="tote_blue", asset_id=f"totes:{_TOTE_BLUE}", op_xy=(0.15, 0.85)),
]

# --- part-count DR ------------------------------------------------------------
# Per-episode random number of parts per class, spawned on the bench. Parts are
# labeled COPIES of graspnet assets: the engine names MuJoCo bodies by
# asset.label, so each instance needs a unique label (duplicates fail compile).
_MAX_PARTS_PER_CLASS = 4
_SCREW_ASSET_ID = "27"                 # graspnet1b "metal screw"
_DRIVER_ASSET_IDS = ["19", "20"]       # blue / red screwdriver, alternated
# Reachable spawn band on the operator-near half of the bench, mapped to the
# canonical frame (world coords, used directly by _spawn_parts).
_PART_LO, _PART_HI = _op_to_canon_region((0.12, -0.20), (0.28, 0.58))
_PART_REGION = dict(x=(_PART_LO[0], _PART_HI[0]), y=(_PART_LO[1], _PART_HI[1]))
_PART_MIN_SEP = 0.10                   # min XY distance between spawned parts
_PART_PLACE_TRIES = 200
# Padding instances are parked far below the floor (z = -10) to keep
# `observation.object_poses` a constant shape; anything under this z is not in
# play and must not count toward the prompt quantities or the reward.
_HIDDEN_Z = -1.0
# Parking slot for the padding instances. Padding is collision-free (see
# `no_collision` in the engine), so the instances can share one spot without any
# contact/impulse. Park them just below the opaque floor — below `_HIDDEN_Z` so
# they still read as padding — and clustered, NOT at z=-10 spread out: MuJoCo's
# shadow map covers the whole `stat.extent`, so padding far from the scene blows
# the extent up (~12 m vs ~5 m) and thins the shadow resolution, which shows up
# as shadow acne flickering on the floor of the VR stream.
_PARK_X, _PARK_Y, _PARK_Z = 0.0, 0.0, -1.2
_PARK_SPACING = 0.0

# Shader params the Isaac engine reads off every ObjectActor (`obj_info.material`).
# The MaterialDR sets these while Task.reset builds the layout, but the totes and
# parts are added AFTERWARDS (in this task's reset), so they'd reach the renderer
# without the attribute and crash it with
# "'ObjectActor' object has no attribute 'material'". MuJoCo never reads it, which
# is why this only ever surfaced at render time. Values match MaterialDR's
# "fixed" branch, keeping the authored look.
_FIXED_OBJECT_MATERIAL = {
    "reflection_roughness_constant": 0.5,
    "metallic_constant": 0.0,
    "specular_level": 0.0,
}

# Target (screw #1) region, operator frame -> canonical.
_TARGET_LO, _TARGET_HI = _op_to_canon_region((0.15, 0.15), (0.22, 0.35))


# Original NVIDIA URLs for the furniture USDs, used as the ISAAC usd_path.
# The downloaded local .usd renders untextured: its materials are referenced with
# paths relative to the asset's original location on the NVIDIA server
# ("../../../../../Materials/Base/..."), which resolve to nothing next to a lone
# local copy — Isaac then logs "could not find module ...Materials::Base..." and
# "Failed to create MDL shade node". Referencing the remote USD makes those
# relative material paths resolve, so the furniture keeps its authored textures.
# (MuJoCo is unaffected: it uses the local mjcf_path/meshes.)
_FURNITURE_USD_URL = {
    "TableTrolley_B02_01":
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
        "DigitalTwin/Assets/Warehouse/Equipment/Carts/TableTrolley_B/TableTrolley_B02_01.usd",
    "MobileShelvingCart_C05_01":
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
        "DigitalTwin/Assets/Warehouse/Equipment/Carts/MobileShelvingCart_C/MobileShelvingCart_C05_01.usd",
}


def _furniture_asset(entry: dict) -> ArticulatedAsset:
    """A 0-joint 'articulated' asset = static furniture attached at a frame.

    Paths are data-relative (resolve_data_path is applied by the engines). The
    MJCF is a full-model attach file (extract_furniture_mesh.py or, for the
    shelf variants, make_shelf_variants.py).
    """
    folder = entry["folder"]
    return ArticulatedAsset(
        uid=entry["name"],
        # remote (textured) for Isaac; falls back to the local copy if unknown
        usd_path=_FURNITURE_USD_URL.get(folder, f"assets/industrial/{folder}/{folder}.usd"),
        mjcf_path=f"assets/industrial/{folder}/{entry['mjcf']}",
        # zero joints: MuJoCo welds it at a frame, Isaac renders it as a plain
        # referenced prim (not a SingleArticulation).
        static=True,
        # These USDs are authored in centimetres (metersPerUnit = 0.01) — the
        # same factor IH_basic.usda carries as `unitsResolve`. MuJoCo already has
        # it baked into the extracted OBJs; Isaac needs it applied on the prim or
        # the trolley comes in 224 m long and swallows the camera.
        usd_scale=0.01,
    )


def _add_furniture(layout: Layout) -> None:
    """Add the warehouse furniture as static 0-joint articulated actors.

    Called after Task.reset has built the layout and run the spatial randomizer.
    Furniture uses `furniture_*` keys, which the spatial randomizer never
    touches, so the poses set here are final. The engine builds these via
    MjSpec.from_file + attach (real mesh collision, Plan A).
    """
    for entry in _FURNITURE:
        key = entry["key"]
        if key in layout.actors:
            continue
        layout.add_articulated_object(key, _furniture_asset(entry))
        actor = layout.actors[key]
        x, y = _op_to_canon_xy(*entry["op_xy"])
        yaw = _op_to_canon_yaw(entry["op_yaw"])
        actor.pose.position = [x, y, 0.0]
        q = t3d.euler.euler2quat(0.0, 0.0, yaw)  # (w, x, y, z)
        actor.pose.quaternion = [float(v) for v in q]


def _add_totes(layout: Layout) -> None:
    """Add the red/blue totes on the bench at fixed spots.

    Added after the spatial randomizer ran, with keys it never touches
    (`tote_*`), at offsets chosen clear of the part-spawn regions so parts
    never spawn underneath a tote. Their color comes from the asset's rgba
    (totes.py color variants), applied by the engine to the MuJoCo geoms.
    """
    table = getattr(layout.scene, "table", None)
    if table is None:
        return
    top = table.pose.position[2] + 0.5 * table.size[2]
    tote_yaw = _op_to_canon_yaw(0.0)  # totes have identity op yaw; map -> pi
    q = t3d.euler.euler2quat(0.0, 0.0, tote_yaw)
    manager = AssetManager.get("totes")
    for entry in _TOTES:
        key = entry["key"]
        if key in layout.actors:
            continue
        asset = manager.load(entry["asset_id"].split(":")[-1])
        layout.add_object(key, asset)
        actor = layout.actors[key]
        actor.set_material(dict(_FIXED_OBJECT_MATERIAL))  # MaterialDR already ran
        x, y = _op_to_canon_xy(*entry["op_xy"])
        stable_z = float(asset.stable_poses[0][2])
        actor.pose.position = [x, y, top + stable_z]
        actor.pose.quaternion = [float(v) for v in q]


def _load_labeled_part(asset_id: str, label: str):
    """A graspnet asset copy with a unique label (and uid) for this instance.

    graspnet's load() builds a fresh asset object per call, so mutating the
    label/uid here never leaks into other instances. Unique labels are required
    because the MuJoCo engine names bodies/meshes after asset.label.
    """
    asset = AssetManager.get("graspnet1b").load(asset_id)
    asset.label = label
    asset.uid = label
    return asset


def _spawn_parts(layout: Layout, replay_parts: list[dict] | None) -> list[dict]:
    """Spawn the per-episode extra parts on the bench (part-count DR).

    Fresh episodes sample n_screws, n_drivers ~ U{1..4}. The spatial-DR-placed
    "target" is always screw #1, so (n_screws - 1) screw copies plus n_drivers
    screwdriver instances are added here with `part_*` keys (ignored by the
    spatial randomizer), rejection-sampled inside _PART_REGION with _PART_MIN_SEP
    clearance from the target, the totes and each other.

    On replay, `replay_parts` (from the recorded state_dict) recreates the exact
    same instances/poses instead of sampling — counts and body names must match
    the recording for the engine scene to line up.

    Returns the spawn spec list to be stored in the task's state_dict.
    """
    import random

    table = getattr(layout.scene, "table", None)
    if table is None:
        return []
    top = table.pose.position[2] + 0.5 * table.size[2]

    if replay_parts is not None:
        for spec in replay_parts:
            asset = _load_labeled_part(spec["asset_id"], spec["label"])
            # Padding instances are identified by their parked (below-floor)
            # position and must stay collision-free on replay too.
            asset.no_collision = spec["position"][2] < _HIDDEN_Z
            layout.add_object(spec["key"], asset)
            actor = layout.actors[spec["key"]]
            actor.set_material(dict(_FIXED_OBJECT_MATERIAL))  # MaterialDR already ran
            actor.pose.position = list(spec["position"])
            actor.pose.quaternion = list(spec["quaternion"])
        return replay_parts

    # Part-Count Domain Randomization:
    # We sample the number of visible parts on the table (1 to 4 each) to vary the episode difficulty.
    n_screws = random.randint(1, _MAX_PARTS_PER_CLASS)
    n_drivers = random.randint(1, _MAX_PARTS_PER_CLASS)

    # Occupied XY spots to keep clear of: target (spatial-placed) and totes.
    occupied: list[tuple[np.ndarray, float]] = []
    target = layout.actors.get("target")
    if target is not None:
        occupied.append((np.array(target.pose.position[:2]), _PART_MIN_SEP))
    for entry in _TOTES:
        tote = layout.actors.get(entry["key"])
        if tote is not None:
            occupied.append((np.array(tote.pose.position[:2]), 0.28))

    def _sample_xy() -> np.ndarray:
        xy = np.array([
            random.uniform(*_PART_REGION["x"]),
            random.uniform(*_PART_REGION["y"]),
        ])
        for _ in range(_PART_PLACE_TRIES):
            if all(np.linalg.norm(xy - pos) >= sep for pos, sep in occupied):
                return xy
            xy = np.array([
                random.uniform(*_PART_REGION["x"]),
                random.uniform(*_PART_REGION["y"]),
            ])
        return xy  # dense episode: accept the last sample rather than fail

    # Dataset Padding Strategy:
    # The LeRobot dataset exporter strictly requires a constant shape for `observation.object_poses`
    # across ALL episodes. To satisfy this while still allowing a variable number of objects on the 
    # table, we ALWAYS spawn exactly _MAX_PARTS_PER_CLASS instances of each object type.
    # The `is_visible` flag determines if the object will be placed on the table (True) or hidden (False).
    to_spawn: list[tuple[str, str, bool]] = []  # (asset_id, label, is_visible)
    
    for k in range(1, _MAX_PARTS_PER_CLASS):  # target already provides screw #1
        to_spawn.append((_SCREW_ASSET_ID, f"metal_screw_{k + 1}", k < n_screws))
        
    for k in range(_MAX_PARTS_PER_CLASS):
        asset_id = _DRIVER_ASSET_IDS[k % len(_DRIVER_ASSET_IDS)]
        base = "blue_screwdriver" if asset_id == "19" else "red_screwdriver"
        label = base if k < len(_DRIVER_ASSET_IDS) else f"{base}_{k // len(_DRIVER_ASSET_IDS) + 1}"
        to_spawn.append((asset_id, label, k < n_drivers))

    spawned: list[dict] = []
    for i, (asset_id, label, is_visible) in enumerate(to_spawn):
        asset = _load_labeled_part(asset_id, label)
        # Padding takes no part in physics (see the parking comment below).
        asset.no_collision = not is_visible
        # Em vez de pegar sempre a pose [0] (que na chave de fenda é vertical e perfeita demais para o MuJoCo derrubar), 
        # sorteamos entre todas as poses de descanso naturais calculadas para aquele asset.
        stable = np.asarray(random.choice(asset.stable_poses), dtype=float)
        
        if is_visible:
            # Place the object normally on the assembly bench
            xy = _sample_xy()
            occupied.append((xy, _PART_MIN_SEP))
            yaw = random.uniform(-np.pi, np.pi)
            z_pos = top + float(stable[2])
        else:
            # Padding Object (Hidden):
            # Parked far below the floor (Z = -10.0), which hides it from the
            # robot's cameras and guarantees it can never trigger the success
            # reward. The downstream Imitation Learning policy will implicitly
            # learn to ignore any object state with a heavily negative Z.
            #
            # IMPORTANT: each padding object gets its OWN parking slot. Stacking
            # them all at the same point makes them spawn deeply interpenetrated,
            # and MuJoCo resolves that with a huge impulse — measured launching
            # them upward at ~143 m/s, straight back THROUGH the workspace where
            # they can strike the robot, the bench and the parts. Since the number
            # of padding objects varies per episode, so did the damage, which is
            # what made performance/behaviour inconsistent across resets.
            xy = np.array([_PARK_X + i * _PARK_SPACING, _PARK_Y])
            yaw = 0.0
            z_pos = _PARK_Z

        # stable orientation composed with a random yaw (same recipe as spatial DR)
        ori = t3d.quaternions.mat2quat(
            t3d.euler.euler2mat(0.0, 0.0, yaw) @ t3d.quaternions.quat2mat(stable[3:7])
        )
        key = f"part_{i}_{label}"
        layout.add_object(key, asset)
        actor = layout.actors[key]
        actor.set_material(dict(_FIXED_OBJECT_MATERIAL))  # MaterialDR already ran
        actor.pose.position = [float(xy[0]), float(xy[1]), float(z_pos)]
        actor.pose.quaternion = [float(v) for v in ori]
        spawned.append(
            dict(
                key=key,
                asset_id=asset_id,
                label=label,
                position=list(actor.pose.position),
                quaternion=list(actor.pose.quaternion),
            )
        )
    return spawned


@TaskRegistry.register("g1_industrial_sorting_teleop")
class G1IndustrialSortingTeleop(Task):
    uid: str = "g1_industrial_sorting_teleop"
    label: str = "G1 TELEOP Industrial Color-Coded Sorting"
    description: str = (
        "An industrial task where the G1 robot sorts screws into a red tote and "
        "screwdrivers into a blue tote, then dispatches the red tote to the left "
        "shelf and the blue tote to the right shelf. Warehouse furniture (bench "
        "and two shelving carts) has real mesh collision."
    )

    metadata: dict[str, Any] = {
        "physics_dt": 0.005,
        "control_hz": 200,
        "render_hz": 50,
        "dr_level": 0,
        "version": 2.3,  # quantity-parametrised prompt + count-based success
        "reward_dt": 0.02,
        "image_dt": 0.033333,
        "need_gravity": True,
        "max_episode_steps": 1200,
    }

    robot_cfg: dict[str, Any] = dict(uid="g1_sonic")

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
        # Quantity-parametrised prompt: {n_drivers}/{n_screws} are filled per
        # episode in reset() from the sampled required counts. Kept in English to
        # match every other task's language conditioning in this repo.
        language=LanguageDRCfg(
            instructions=[
                "put {drivers} in the blue tote and {screws} in the red tote, "
                "then place the blue tote on the right shelf and the red tote "
                "on the left shelf.",
            ]
        ),
        # Parts on the bench: the target is the metal screw (= screw #1, placed
        # by the spatial DR). ALL other parts — extra screws and the
        # screwdrivers — come from the part-count DR in _spawn_parts (1..4 per
        # class, per episode), as labeled asset copies. No DistractorDRCfg: the
        # distractor pipeline can't spawn repeated assets (the engine names
        # bodies by asset.label, so instances need unique labels).
        target=TargetDRCfg(asset_id="graspnet1b:27"),  # "metal screw"
        spatial=SpatialDRCfg(
            spatial_mode="random",
            # CANONICAL spawn: -x side, yaw 0 (facing +x toward the bench). The
            # WBC teleop stack (leg policy + hand retargeting) is calibrated for
            # this heading; spawning yawed 180 broke VR responsiveness, so the
            # scene is arranged around this pose (operator layout mapped onto the
            # canonical robot). MUST match _CANON_ROBOT_XY.
            robot_region=Box(low=[-0.63, 0.25, 0.0], high=[-0.65, 0.25, 0.0]),
            target_region=Box(low=list(_TARGET_LO), high=list(_TARGET_HI)),
            target_stable_indices=[0],
            target_rotate_z=Box(low=np.pi, high=np.pi),
            obj_surface_map={
                "target": "table",
            },
        ),
        camera=CameraDRCfg(cam_id="franka_camera"),
        scene=TabletopSceneDRCfg(
            scene_manager="warehouse",
            room_choices=["warehouse:default"],
            scene_mode="fixed",
            table_size=Box(low=[0.8, 2.0, 0.1], high=[0.8, 2.0, 0.1]),
            # bench mapped into the canonical frame (must match _BENCH_CANON_XY);
            # the trolley MESH is yawed pi in _FURNITURE, the box is symmetric.
            table_position=Box(low=list(_BENCH_CANON_XY), high=list(_BENCH_CANON_XY)),
            table_height=Box(low=_BENCH_TOP, high=_BENCH_TOP),
            rotation_z=Box(low=0.0, high=0.0),
            enable_table2=False,
        ),
        # Lighting is the ONLY visual randomization at the rendering stage:
        # widened colour temperature (warm 3000K -> cool 8000K) and intensity so
        # episodes differ in illumination while textures stay original.
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
        # Rendering stage: keep the ORIGINAL look of furniture and objects —
        # "fixed" stops the per-episode texture draw for the table/ground and
        # pins the shader params (roughness/metallic/specular) to their defaults
        # instead of sampling them. Furniture, totes and parts already render
        # with their own USD materials (the engine only overrides table/ground),
        # so this keeps the whole scene on its authored textures. Visual variety
        # comes from LightingDRCfg below.
        material=MaterialDRCfg(material_mode="fixed"),
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
        *args,
        **kwargs,
    ):
        self._instruction = None
        self._target = None
        self._layout = None
        self._extra_parts: list[dict] = []
        self._required_counts: dict[str, int] = {"screws": 1, "drivers": 1}

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
    def required_counts(self) -> dict[str, int]:
        """How many of each class this episode asks for ({"screws", "drivers"}).

        Sampled per episode in [1, n_spawned]; drives both the prompt and the
        success thresholds. Read by the teleop CLI to build the VR HUD.
        """
        return dict(self._required_counts)

    @property
    def tote_red(self) -> Actor:
        return self.layout.actors["tote_red"]

    @property
    def tote_blue(self) -> Actor:
        return self.layout.actors["tote_blue"]

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
        _add_furniture(self.layout)
        _add_totes(self.layout)
        # Part-count DR: fresh episodes sample counts; replay recreates the
        # recorded instances exactly (body names must match the recording).
        replay_parts = None
        if options is not None and options.get("state_dict") is not None:
            replay_parts = options["state_dict"].get("extra_parts")
        self._extra_parts = _spawn_parts(self.layout, replay_parts)
        self._target = self.layout.actors.get("target")

        # Quantity DR: how many of each class the operator is asked to sort this
        # episode. Sampled in [1, n_spawned] per class (never 0, may be all of
        # them). Replayed from the recorded state_dict so the prompt AND the
        # success threshold match the recording.
        screws, drivers = self._part_labels()
        replay_counts = None
        if options is not None and options.get("state_dict") is not None:
            replay_counts = options["state_dict"].get("required_counts")
        if replay_counts is not None:
            self._required_counts = dict(replay_counts)
        else:
            import random

            self._required_counts = {
                "screws": random.randint(1, max(1, len(screws))),
                "drivers": random.randint(1, max(1, len(drivers))),
            }

        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        language_template = lang_dr(self.metadata.get("split", "train"))
        def _qty(n: int, noun: str) -> str:
            return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

        self._instruction = language_template.format(
            drivers=_qty(self._required_counts["drivers"], "screwdriver"),
            screws=_qty(self._required_counts["screws"], "screw"),
        )
        self.reward = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update(
            {
                "tote_red_uid": _TOTE_RED,
                "tote_blue_uid": _TOTE_BLUE,
                # part-count DR spawns, so replay recreates the same bodies
                "extra_parts": self._extra_parts,
                # quantity DR: replay must reuse the same prompt AND thresholds
                "required_counts": dict(self._required_counts),
            }
        )
        return state_dict

    # ------------------------------------------------------------------ checks

    def _part_labels(self) -> tuple[list[str], list[str]]:
        """(screw labels, screwdriver labels) that are actually IN PLAY.

        The spawner always creates _MAX_PARTS_PER_CLASS instances per class so the
        recorded `observation.object_poses` keeps a constant shape across episodes,
        parking the unused ones far below the floor. Those padding instances are
        filtered out here so they never inflate the sampled prompt quantities nor
        the reward thresholds.

        Order matters in the tests: 'screwdriver' contains 'screw', so drivers
        are matched first.
        """
        screws: list[str] = []
        drivers: list[str] = []
        for name, actor in self.layout.actors.items():
            if name != "target" and not name.startswith("part_"):
                continue
            if actor.pose.position[2] < _HIDDEN_Z:  # padding instance, not in play
                continue
            label = str(actor.asset.label).lower()
            if "screwdriver" in label:
                drivers.append(str(actor.asset.label))
            elif "screw" in label:
                screws.append(str(actor.asset.label))
        return screws, drivers

    @staticmethod
    def _contact_pairs(mj_model, mj_data) -> set[frozenset]:
        """Set of unordered body-name pairs currently in contact."""
        pairs: set[frozenset] = set()
        for i_contact in range(mj_data.ncon):
            contact = mj_data.contact[i_contact]
            b1 = mj_model.body(mj_model.geom(contact.geom1).bodyid).name
            b2 = mj_model.body(mj_model.geom(contact.geom2).bodyid).name
            if b1 != b2:
                pairs.add(frozenset((b1, b2)))
        return pairs

    @staticmethod
    def _body_xy(mj_model, mj_data, name: str):
        """Live world XY of a body, or None if the body doesn't exist."""
        try:
            bid = mj_model.body(name).id
        except Exception:
            return None
        return np.array(mj_data.xpos[bid][:2])

    def _count_in_tote(self, mj_model, mj_data, pairs, tote: str, part_labels: list[str]) -> int:
        """How many of the listed parts are in the tote.

        A part counts when it touches the tote AND sits within its XY footprint.
        Uses live body positions from mjData (not the layout's initial poses).
        """
        tote_xy = self._body_xy(mj_model, mj_data, tote)
        if tote_xy is None:
            return 0
        count = 0
        for part in part_labels:
            if frozenset((tote, part)) not in pairs:
                continue
            part_xy = self._body_xy(mj_model, mj_data, part)
            if part_xy is None:
                continue
            if np.all(np.abs(tote_xy - part_xy) <= _IN_TOTE_XY_TOL):
                count += 1
        return count

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info, *args, **kwargs)
        return reward >= self.success_criteria

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        """0.25 per condition; 1.0 (success) only when the full sort is done, with
        the per-episode quantities the prompt asked for:
        >=Y screws in the red tote, red tote on the LEFT shelf,
        >=X screwdrivers in the blue tote, blue tote on the RIGHT shelf.
        (X/Y = self._required_counts; "at least", so extra parts don't invalidate.)
        """
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            self.reward = 0.0
            return self.reward

        mj_model = mujoco_env.mjModel
        mj_data = mujoco_env.mjData
        pairs = self._contact_pairs(mj_model, mj_data)
        screws, drivers = self._part_labels()

        n_screws_in_red = self._count_in_tote(mj_model, mj_data, pairs, _TOTE_RED, screws)
        n_drivers_in_blue = self._count_in_tote(mj_model, mj_data, pairs, _TOTE_BLUE, drivers)
        screws_ok = n_screws_in_red >= self._required_counts["screws"]
        drivers_ok = n_drivers_in_blue >= self._required_counts["drivers"]
        red_on_left = frozenset((_TOTE_RED, _SHELF_LEFT)) in pairs
        blue_on_right = frozenset((_TOTE_BLUE, _SHELF_RIGHT)) in pairs

        self.reward = 0.25 * (
            int(screws_ok) + int(red_on_left) + int(drivers_ok) + int(blue_on_right)
        )
        return self.reward

    def preload_objects(self) -> list[Actor]:
        """Preloads all assets required by the task."""
        asset_manager = AssetManager.get("graspnet1b")
        return [ObjectActor(asset=asset) for asset in asset_manager]
