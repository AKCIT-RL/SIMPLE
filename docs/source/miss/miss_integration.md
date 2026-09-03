# Miss Integration

This document describes the integration of the user's real "Miss" platform into SIMPLE as a new
embodiment, on branch `feat/add-widowx`. Miss is a Clearpath Jackal (j100) wheeled base carrying
a single Trossen ViperX arm (the stock Interbotix vx300s, not Aloha's fork), a stock Interbotix
gripper, and a 2-DOF pan/tilt gimbal carrying an Intel RealSense D455. Its hardware description
lives at `viperx/misskal_hardware/` (an untracked local clone of the user's separate ROS2
project, unrelated to this repo's own git history). Only the arm, gripper, and gimbal are
functional in this integration; the wheeled base is modeled in full but not yet driven (see
"Known limitations" below). The robot is registered as uid `miss`
(`src/simple/robots/miss.py`); no task/environment exists for it yet (deliberately deferred, see
"Known limitations").

## Compatibility with ViperX

Before building Miss, the user's arm was compared directly against `ViperX`
(`docs/source/viperx/viperx_integration.md`, extracted from Aloha's left arm). Miss's arm
resolves, via the `interbotix_ros_manipulators` submodule, to the **stock, upstream Interbotix
`vx300s.urdf.xacro`** — not a fork. All six arm joint limits are numerically identical to Aloha's
own vx300s fork's, and the shared meshes are byte-identical (`md5sum` confirmed on
`base.stl`/`shoulder.stl` against `data/robots/aloha/assets/`). The gripper genuinely differs:
Aloha/ViperX use a custom 2-independent-finger parallel gripper with an integrated D405 wrist
camera; Miss's stock gripper is a single mimic-jointed prismatic pair (`left_finger` driving,
`right_finger` mirroring via `<mimic multiplier="-1">`) plus a cosmetic continuous
`gripper`→`gripper_prop_link` joint, and has no wrist camera at all.

## Assets: mechanically flattened from real ROS xacro, not hand-authored

Miss's hardware description is deeply nested ROS2 xacro (`misskal_description`,
`clearpath_platform_description`, `clearpath_sensors_description`, `interbotix_xsarm_descriptions`,
cross-referencing each other via `$(find package)`), unlike every other robot integrated into
SIMPLE so far (Franka/Aloha/WidowX AI/ViperX all started from an already-flat, hand-authored
MJCF). Rather than hand-transcribing this, the actual entry point,
`misskal_description/urdf/jackal.urdf.xacro` (**not** `misskal.urdf.xacro`, which is an empty
0-byte placeholder in this checkout — found by directly inspecting the file, not assumed), was
flattened programmatically:

1. Installed the `xacro` Python package (`uv pip install xacro`) and monkeypatched its package
   resolver (`ament_index_python.packages.get_package_share_directory`, which normally requires a
   real ROS2/ament install) to map package names directly onto this checkout's actual submodule
   directories — no ROS2 install needed.
2. Two packages referenced by the real camera macros (`realsense2_description`, providing the
   D455/D435i camera-body xacro macros) are not vendored anywhere in this checkout. Minimal
   no-op stub macros (`usb_plug`, `d435i_imu_modules`) were authored just so `xacro:include`
   resolves — both are only ever invoked behind an `add_plug:=false`-gated branch that is never
   actually expanded, so the stubs never needed real bodies.
3. `xacro.process_file(...)` on `jackal.urdf.xacro` with `hardware_type="fake"` (avoids pulling
   in real/Gazebo ROS-control plugin blocks) produced a single, fully-resolved URDF with **zero**
   unresolved package references — the entire platform (39 links: chassis, 4 wheels, lidar, both
   cameras, the arm, the gripper, the gimbal) in one file.
4. Post-processed: stripped `<gazebo>`/`<ros2_control>`/`<transmission>` (MuJoCo's URDF importer
   ignores these anyway), dropped the two camera-body links whose only geometry pointed at the
   unavailable `realsense2_description` meshes (`d435i_link`, `d455_link` — their *mount* links,
   which carry the real pose, are untouched), and rewrote every `package://<pkg>/...` mesh URI to
   a locally-copied file. Every real mesh referenced (20 files, from
   `clearpath_platform_description`, `interbotix_xsarm_descriptions`, and `misskal_description`)
   was copied into `data/robots/miss/` directly alongside `miss.urdf`/`miss.xml` — flat, not
   nested under an `assets/` subdirectory, because MuJoCo's URDF importer strips directory
   components from `<mesh filename>` paths (confirmed empirically: mesh loading failed with an
   `assets/` subdirectory, succeeded once files were moved flat).

**A real bug was found (and fixed only in our converted copy, not the user's source) during this
process:** `jackal.urdf.xacro`'s `vx300s_joint` declares `<child link="misskal/vx300s_base"/>`,
but the actual `vx300s.urdf.xacro` include (using its own default args) produces a link named
`vx300s/base_link` — these never match, so as written the arm is not actually attached to the
rest of the tree. MuJoCo's URDF parser caught this immediately ("joint parent or child missing").
Fixed by correcting the `<child>` reference in `data/robots/miss/miss.urdf`; the user's own
`jackal.urdf.xacro` (currently being edited, uncommitted) likely needs the same fix, or to pass
matching `robot_name`/`base_link_frame` args to the `vx300s.urdf.xacro` include.

**Naming departs from ViperX's convention on purpose.** ViperX kept Aloha's `left_`-prefixed
joint names (needed there to disambiguate two arms). Miss has only one arm, so its joint names
are the real hardware names as produced by the stock xacro and used by
`misskal_control/config/control.yaml` (`waist`, `shoulder`, `elbow`, `forearm_roll`,
`wrist_angle`, `wrist_rotate`, `gripper`, `left_finger`, `right_finger`) — geometry/limits are
still numerically identical to ViperX's, just not literally spliced from `viperx.xml`.

`data/robots/miss/miss.urdf` is curobo's own kinematics source (loaded directly by
`CuRoboMixin`). `data/robots/miss/miss.xml` is a native-MJCF dump of the same tree
(`mujoco.MjSpec.from_file('miss.urdf').to_xml()`), used by the MuJoCo engine — see "MuJoCo-specific
fixes" below for why a plain URDF `<mujoco>` extension block does not work as a way to add
actuators directly to the `.urdf`.

## MuJoCo-specific fixes required after conversion

Three issues were found and fixed only by actually loading the converted asset in MuJoCo, not by
reading the source xacro:

1. **No actuators, no equality constraint.** MuJoCo's URDF importer creates zero actuators and
   does not honor URDF `<mimic>` (`right_finger`'s `<mimic joint="left_finger" multiplier="-1"/>`
   did not survive xacro's flatten of `jackal.urdf.xacro` at all — confirmed against the raw
   xacro output before any of this integration's own post-processing touched it, so this was
   xacro's own behavior, not something we broke). An embedded `<mujoco>` extension block inside
   the `.urdf` (the documented mechanism for adding MJCF-only elements to a URDF import) was
   tried first and empirically does **not** work with this MuJoCo version/setup — confirmed via
   `mujoco.MjSpec.from_file(...).actuators`/`.equalities` staying empty regardless. The working
   fix: dump the loaded spec to native MJCF (`spec.to_xml()`) and add `<actuator>`/`<equality>`
   directly to that file, the same way every other hand-authored robot MJCF in this repo already
   does it. The `<mimic>` tag was also restored directly in `miss.urdf` itself (curobo's own URDF
   kinematics *does* natively understand `<mimic>`, unlike MuJoCo's importer).

   **Follow-up gap found later (fingers appearing asymmetric during real gripper commands):** the
   `<equality>` alone was not sufficient. `right_finger` has no actuator of its own by design, and
   nothing in `Miss.setup_control()` (`src/simple/robots/miss.py`) ever set its initial `qpos` --
   it started every episode at MuJoCo's implicit default `0.0`, which sits *outside*
   `right_finger`'s own joint range (`[-0.057, -0.021]`, mirrored from `left_finger`'s
   `[0.021, 0.057]`) and inconsistent with the equality's target for whatever `left_finger` had
   just been initialized to. The compliant equality constraint (a soft pull, not a hard teleport)
   then had to fight both its own out-of-range joint limit and the mismatch simultaneously, with no
   actuator of its own to help -- visibly, `left_finger` would snap straight to its
   actuator-commanded target while `right_finger` lagged or sat at the wrong position. Fixed by
   explicitly mirroring `right_finger`'s `qpos`/`qvel`/`qacc` from `left_finger`'s right after
   `set_initial_qpos()` runs, in `setup_control()`. Verified directly: immediately after reset,
   `left_finger.qpos + right_finger.qpos == 0` exactly; after settling from an open/close command,
   the sum drifts by only ~0.00045 (consistent with the small P-only-controller steady-state droop
   already documented for other joints below), not the multi-centimeter mismatch seen before the
   fix.

   **A second, unrelated bug was found in the same investigation**: `ParallelGripperEEFController
   .open_gripper()`/`.close_gripper()` (`src/simple/robots/controllers/eef.py`) hardcoded
   `ctrl = 0.0205` / `ctrl = 0.` for every robot using this controller -- values tuned for
   ViperX/Aloha's `general`-actuator gripper (unclamped, so any ctrl value is accepted), not for
   Miss's `left_finger`, which uses a real `<position>` actuator with `ctrlrange="0.021 0.057"`.
   Both hardcoded values fell outside that range and silently clamped to the same boundary
   (`0.021`), making Miss's "open" and "close" commands produce the *identical* joint position.
   Fixed by adding optional `open_qpos`/`close_qpos` lists to `ParallelGripperEEFControllerCfg`
   (defaulting to `None`, which preserves the old hardcoded values exactly -- confirmed ViperX's
   own open/close behavior is bit-for-bit unchanged) and setting Miss's controller_cfg to its own
   real range (`open_qpos=[0.057]`, `close_qpos=[0.021]`).
2. **Underdamped position actuators (persistent oscillation, not settling).** The stock xacro's
   generic `<dynamics damping="0.1" friction="0.1"/>` on every arm joint is a placeholder, not a
   tuned simulation value — confirmed by tracing a single joint's qpos over time: commanding
   `shoulder` to -1.57 rad with the URDF-derived damping produced a sustained oscillation between
   roughly -0.97 and -1.83 rad that never converged, not just a slow settle. Fixed by copying
   Aloha's own already-tuned per-joint `damping`/`armature`/`frictionloss`/`actuatorfrcrange`
   values (same physical vx300s hardware) onto Miss's arm joints in `miss.xml`. The gimbal has no
   such reference value anywhere upstream; `damping="2.0"` was picked empirically (via
   `scripts/test_miss_gimbal_arm.py`) to reach reasonable settling, not sourced or rigorously
   tuned.
3. **Pose-invariant self-collisions from the mount geometry.** At the arm's stock "Home" pose
   (all-zero), `mujoco.mj_forward` reported the gimbal's neck/head and the arm's shoulder link
   colliding with the static chassis/mount structure. A full sweep of `waist` over `[-1, 1]` rad
   and `head_pan`/`head_tilt` over their entire range never cleared these three contacts — i.e.
   not a bad-pose problem, a structural one. Read as the gimbal's pivot bracket and the arm's
   shoulder sitting flush against their own mounting bracket in the source CAD (a real
   bearing/clearance gap on the physical hardware that a rigid mesh-vs-mesh check can't
   represent) — the same class of "flush-mount" pseudo-collision already documented and excluded
   for Aloha's own base/shoulder link pair. Fixed with three `<contact><exclude>` pairs in
   `miss.xml`.

## Named presets

`src/simple/robots/miss.py` exposes `ARM_PRESETS`, `GRIPPER_PRESETS`, `GIMBAL_PRESETS`. The arm
and gripper presets are Interbotix's own stock vx300s MoveIt SRDF `group_state`s
(`interbotix_xsarm_moveit/config/srdf/vx300s.srdf.xacro`: arm **Home**/**Upright**/**Sleep**,
gripper **Grasping**/**Released**/**Home**) — real, sourced values. Each arm × gripper × gimbal
combination was cross-checked against the full Miss asset (`mujoco.mj_forward`/`d.ncon`):
**Home** and **Upright** are collision-free for every combination tried; **Sleep** is **not** —
folding the arm into Interbotix's stock Sleep configuration collides the gripper with the chassis
for every gripper/gimbal combination, a genuine pose-dependent collision specific to how the arm
is mounted on this platform (unlike the pose-invariant contacts already excluded above). `Miss`'s
own class-level `init_joint_states`/`controller_cfg` use Home (arm) + Home (gripper), the
combination confirmed collision-free and used as the spawn default.

No equivalent presets exist anywhere in `misskal_hardware` for the gimbal (only its raw joint
limits) — `GIMBAL_PRESETS` (`stow`, `look_down`, `look_down_more`) are authored fresh for SIMPLE,
not sourced, and paired with `gimbal_cam`'s own unverified orientation (see below) — treat as a
starting point to iterate on visually, not validated values.

**The class-level default pose was later replaced** with a user-chosen "desk reach" configuration,
which then went through several more rounds of correction (see "Arm mount rotated 180°" below for
why) before settling on its current value: arm `[0.0, -1.60, 0.20, -0.0, 1.20, -0.0]`, gripper open
(`0.057`), gimbal `[head_pan, head_tilt] = [0.0, -1.9025]` — confirmed collision-free the same way
(`d.ncon == 0`, plus a successful curobo self-collision `IKSolver` query) before adopting it. Also
registered as `ARM_PRESETS["desk_reach"]` / `GIMBAL_PRESETS["look_at_desk"]`. **Note the gimbal
ordering is easy to transpose**: the interactive/curobo scripts (below) take gimbal input as
`[pitch, yaw]` (pitch→`head_tilt`, yaw→`head_pan`), the *reverse* of `gimbal_joint_names`'s own
`["head_pan", "head_tilt"]` order — `gimbal_init_qpos` must be written in the latter order.

**Earlier `desk_reach` values, superseded (kept here for the historical record, not to be
reused):**
- `[3.1377, -1.4941, 1.5015, -0.0, -1.0907, -0.0054]` — the original choice, `waist=-3.14` only
  ~0.0016 rad from the joint's own `±π` range limit. This is what motivated the arm-mount rotation
  below once it was linked to a real motion-planning symptom.
- `[3.1377, -1.941, 0.5015, -0.0, 0.907, -0.0054]` — an intermediate attempt (before the mount
  rotation) that **collided**: `mujoco.mj_forward` reported `vx300s/upper_arm_link` and
  `vx300s/upper_forearm_link` penetrating the chassis (`chassi.stl`, welded to `world` since
  `base_link` has no joint) by up to 7.3cm, and curobo's own self-collision `IKSolver` query failed
  for the same configuration. Rejected before being adopted, per the same
  collision-check-before-commit discipline as every preset in this doc.
- `[3.14, -1.60, 0.20, -0.0, 1.20, -0.0]` — adopted, collision-free, immediately before the mount
  rotation (`waist=3.14`, ≈0.0016 rad from the *opposite* limit — the rotation below moved this to
  `waist=0.0`, unrelated to the collision-freeness of the other five joints, which stayed the same
  across both).

![Miss at the Home preset](media/miss_home_pose.png)
![Miss at the Upright preset](media/miss_upright_pose.png)

## Visualization/test script

`scripts/test_miss_gimbal_arm.py` drives Miss directly (no `gym.make(...)`, since no task/env
exists yet) through a fixed sequence of arm × gripper × gimbal presets via position actuators,
reporting per-preset settling error. Unlike the curobo-IK-based test scripts for other robots, no
Cartesian planning is involved — this only needs to move between named joint-space presets.
`--no-headless` opens the MuJoCo viewer to watch it move; `--headless` (default) just reports
settling error, exit code 0 iff every preset in the sequence settles within `--pos-tol`. A
nonzero steady-state error against joint friction/gravity is expected from these P-only position
actuators (matching every other robot's controller in this codebase, none of which use integral
gain) — `--pos-tol` defaults to a loose `0.05` for this reason, not because any specific bound
was derived; "Upright" carries the largest residual droop of the presets tried (heaviest gravity
moment at that arm extension).

`scripts/miss_interactive_control.py` and `scripts/test_miss_curobo_ik.py` were added afterward:
the former is a terminal-driven live control loop (type `arm = [...]`/`gimbal = [pitch, yaw]`/
`gripper open|closed`, watch it move, read back the settled qpos in the same format — meant for
finding new named positions interactively) with physics/viewer running on a background thread
while stdin is read on the main thread; the latter validates curobo IK for the arm specifically
(6/6 waypoints converge, generated from `Miss.fk()` at representative joint configs, not
hand-picked) while driving the gimbal directly via plain actuator commands, deliberately outside
curobo (the gimbal is not part of the arm's kinematic chain and is not in `curobo/miss.yml`'s
cspace). Both needed the same P-only-controller steady-state-droop tolerance adjustment already
described above (a few cm of Cartesian error / up to ~0.04 rad of joint error, not a bug).

## curobo motion planning

`data/robots/miss/curobo/miss.yml` / `curobo/spheres/miss.yml`. The arm-link spheres are reused
verbatim from `data/robots/viperx/curobo/spheres/viperx.yml` (byte-identical meshes), re-keyed
from `left_*_link` to `vx300s/*_link`. The gripper-link spheres (`gripper_link`,
`gripper_prop_link`, `left_finger_link`, `right_finger_link`) are a **first-pass, single-sphere
bounding fit per link**, computed directly from real mesh vertex data (not hand-guessed), but not
a proper multi-sphere fit against MJCF collision primitives the way
`data/robots/viperx/generate_spheres.py`/`data/robots/widowx_ai/generate_spheres.py` do — treat
as a coarse, likely over-approximating placeholder. `self_collision_ignore` follows the exact
same convention already validated for ViperX/Aloha: every arm/gripper link ignores every other
arm/gripper link (adjacent-link contact at a serial arm's own joints is expected, not a real
collision). FK and a trivial self-collision `IKSolver` query (seed = target = the Home retract
config) both succeed.

`get_grasp_pose_wrt_robot`'s **rotation** (`T_grasp_ee`) is left as an identity transform,
explicitly not re-derived. Unlike ViperX (which reused Aloha's already-validated
`*_gripper_site` frame convention verbatim, since it's the identical hardware), Miss's stock
`ee_gripper_link` frame convention has no prior derivation to reuse and has not been independently
verified against GraspNet's own convention the way WidowX AI's was — do not trust this for real
grasp planning without repeating that derivation.

**The translation half of the same function had a confirmed, reproducible bug, since fixed**: it
ignored `robot_eef_offset` (`0.0385`, `fingers_link` → `ee_gripper_link`, already defined on the
class) entirely, commanding curobo to put `ee_gripper_link` itself — not the actual finger contact
point 3.85cm further along the approach axis — at the object's grasp point. Confirmed directly:
with the identity transform, real grasp candidates landed only ~1.4cm above `table2`'s real
surface (see "Object placement" below for `table2`), and every motion-planning attempt failed with
`MotionGenStatus.IK_FAIL` — despite the same poses being trivially IK-reachable in isolation
(collision-free `robot.ik()` succeeded for them), proving this was a real collision against the
desk, not an unreachable pose. Reproduced the fix directly against the production planning call
(`CuRoboPlanner.batch_plan_for_approach`): applying `T_ee_hand[:3,3] = [-robot_eef_offset, 0, 0]`
(matching ViperX/Aloha's own convention) moved 3 tested candidates from `FAILED` to `SUCCESS`,
each gaining the expected ~4cm of clearance above the desk.

## Grounding the scene: MuJoCo's synthetic floor vs. the robot's real footprint

After the `table`/`table2` fix above, the robot and desk were correctly positioned *relative to
each other*, but the whole scene still visually looked like it was floating above the checkered
floor. Root cause: MuJoCo has **no real HSSD room geometry at all** (confirmed by reading
`src/simple/engines/mujoco.py`) — the only floor-like thing in the MuJoCo backend is a synthetic
`mjGEOM_PLANE` groundplane, positioned by:
```python
z_minus = self.task.layout.scene.table.pose.position[2] + 0.5 * self.task.layout.scene.table.size[2]
if hasattr(self.task.robot, "z_offset"):
    z_minus += self.task.robot.z_offset - self.robot_z
```
This uses the *abstract* `"table"` actor's own intended height as "where the floor is" — for
`table_height=0.70`, `z_minus≈0.75`, putting the groundplane a full 70cm below Miss's actual
wheel-contact plane (`world Z=0`), completely decoupled from where the robot's wheels really are.
`Vega1`'s existing `z_offset` mechanism (a *constant* shift) can't fix this, because the gap here
is `table_height`-dependent, not constant.

Fixed with a new opt-in robot attribute, `floor_grounded: bool = True` (`src/simple/robots/
miss.py`), and a matching branch in the engine (`src/simple/engines/mujoco.py`): if the robot
declares it, `z_minus = 0.0` unconditionally — the groundplane goes to literal world Z=0, matching
`_ROBOT_Z_OFFSET`'s own design (Miss's wheels always touch Z=0 regardless of `table_height`). Every
other robot is untouched (`getattr(..., "floor_grounded", False)` only fires for one that opts in)
— regression-tested directly against ViperX's own groundplane placement (identical before/after).
Verified by direct MuJoCo geom introspection (`ground` geom lands at exactly `pos=[0,0,-0.]`,
matching the wheel-bottom contact point) and a render showing the wheels resting correctly on the
checkered floor.

## GSNet grasp-pose lookup: from a Z-matching heuristic to real `stable_idx` plumbing

`GSNet.load_cached_grasps` (`src/simple/grasps/gsnet.py`) identifies which cached grasp set to use
by finding which `asset.stable_poses[i]` the target is currently resting in — originally via an
**exact match** (`atol=1e-8`) between the target's absolute world Z and `stable_poses[i][2]`. This
implicitly assumed the object's resting surface is at world Z=0 (`surface_height == 0`) — true for
every task before Miss by coincidence, not a real invariant — and raised `"No matching stable pose
found"` for Miss's nonzero `table_height`, blocking `datagen` from completing a single episode.

A first fix attempt (residual-based disambiguation: pick the stable pose whose Z, plus an unknown
`surface_height`, best explains the target's Z) was **tried, tested, and rejected**: the banana
asset (`graspnet1b:5`) has 7 stable poses whose raw Z values differ by only ~1.4mm — far smaller
than any realistic `surface_height` — so "smallest non-negative residual" ends up just ranking
candidates by their own raw Z, unrelated to which one is actually true. Verified this concretely
with a synthetic test before discarding the approach.

**The actual fix**: thread the real, already-known `stable_idx` (picked once by `SpatialDR
._random_place_one_object`, `src/simple/dr/spatial.py`, at placement time) forward to
`GSNet.load_cached_grasps(stable_idx=...)` directly, instead of ever guessing it back from Z. Added
`Object.stable_idx: Optional[int] = None` (`src/simple/core/object.py`), set it wherever
`_random_place_one_object` commits a placement (and restored from `_inner_state` on the
cache-hit/replay path, so it survives a cached re-placement too), and pass
`stable_idx=getattr(target_actor, "stable_idx", None)` in `MotionPlannerAgent`'s call
(`src/simple/agents/mp.py`) — `None` when unset preserves every existing task's old (working, since
`surface_height==0`) behavior exactly. Verified end-to-end: after a real `env.reset()`, the
target's `stable_idx` matches the grasp set `GSNet` actually resolves, with zero Z-matching
involved.

## Object placement decoupled from desk placement

The banana's spawn position was originally reused from `table2`'s own center
(`table_distance`/`table_angle`) — but `table2` is `2.0m` deep, so this put the banana ~1.75m from
the robot, well past anything the arm had ever reached in FK/IK checks (<1m). Added independent
`object_distance`/`object_angle` constructor parameters (same "direction from robot origin"
convention as `table_distance`/`table_angle`, but never coupled to it), defaulting the banana
close to the robot's own reach rather than the table's center. A construction-time check warns
(does not fail) if a chosen `object_distance`/`object_angle` combination would land outside
`table2`'s own footprint, rather than silently spawning the banana off the desk.

## First real `datagen` run: `IK_FAIL` traced to `get_grasp_pose_wrt_robot`, not reachability

With the fixes above landed, a real `datagen` run (`simple/MissTabletopGraspMP-v0`,
`--ignore-target-collision`, 1 episode) got past grasp synthesis with no crash for the first time —
but every attempt failed motion planning with `MotionGenStatus.IK_FAIL`. Diagnosed methodically
rather than by re-tuning distances blindly: single-target, no-collision `robot.ik()` succeeded for
the exact same candidate poses (proving they're kinematically reachable), while the real,
collision-aware `CuRoboPlanner.batch_plan_for_approach` failed for the same poses — isolating the
cause to *collision*, not reachability, which led directly to the `get_grasp_pose_wrt_robot`
offset bug documented above (the actual, root-caused fix). A follow-up run with that fix applied
got through the same call with no `IK_FAIL` at all.

**Replaying one of these recorded smoke-test episodes** (e.g. the `miss_smoke`/`miss_smoke1`
.. `miss_smoke5` directories under `data/datagen/`, left over from this round of `datagen` runs):
`replay`'s `--data-dir` must point at the actual LeRobot dataset root `datagen` wrote to, which is
**not** the `--save-dir` value passed to `datagen` itself but that path plus `<env_id>/level-<dr_level>`
(`LerobotRecorder.__init__`, `src/simple/envs/lerobot.py`: `dataset_root_dir =
f"{root_dir}/{task_name}/level-{dr_level}"`, `dr_level` defaults to `0`) — confirmed directly
against one of the on-disk `miss_smoke*` directories, e.g.:
```
data/datagen/miss_smoke5/simple/MissTabletopGraspMP-v0/level-0/{meta,data,videos}/...
```
So to replay it:
```bash
replay simple/MissTabletopGraspMP-v0 \
  --sim-mode mujoco \
  --data-dir data/datagen/miss_smoke5/simple/MissTabletopGraspMP-v0/level-0 \
  --no-headless
```
(drop `--no-headless` for a headless run; `env_id` is still the positional argument even though
the dataset root already encodes it, since it's also used as the HF `repo_id` when loading via
`LeRobotDataset(repo_id=env_id, root=data_dir)`). Swap `miss_smoke5` for any of the other
`miss_smoke*`/`miss_smoke` directories to replay a different recorded run.

## LeRobot dataset action-shape mismatches

Two separate `ValueError: The feature 'action' of shape '(N,)' does not have the expected shape
'(M,)'` errors surfaced once motion planning itself started succeeding — the same class of bug
already documented and fixed for WidowX AI/ViperX in `src/simple/envs/lerobot.py`, here for two new
reasons specific to Miss:
- `SingleArmGimbalBinaryEEFController.action_space` (`src/simple/robots/controllers/combo.py`)
  originally declared `{"arm": 6, "gimbal": 2}` = 8 dims — but `MotionPlannerAgent` never actually
  drives the gimbal through `target_qpos` during this grasp task (gimbal_qpos, when set at all,
  goes through a separate `ActionCmd` key), so the *recorded* action was always 7-wide (6 arm + 1
  gripper). Fixed by making the action space arm-only (`{"arm": 6}`), matching what's actually
  commanded — a deliberate choice (recorded dataset actions describe only what the policy actually
  controls in this task), not the only option (an alternative, recording the gimbal's live-but-
  unchanging qpos every frame, was considered and rejected as adding a constant, uninformative
  dimension).
- Once the schema became arm-only (6-wide), Miss's own `left_finger` joint still fell through
  `lerobot.py`'s generic finger-dedup branch and got appended, making the recorded vector 7-wide
  again. Fixed with an explicit `"miss" in self.robot.uid and jname == "left_finger": continue`
  skip, mirroring the existing `"viperx" in self.robot.uid` skip immediately above it in the same
  file.

## Arm mount rotated 180° to move the waist joint's wrap-around limit

While testing the (then-current) `desk_reach` pose interactively, the user observed the arm
frequently sweeping the `waist` joint almost a full 360° ("the long way around") when planning
grasp approaches — with `waist` range `[-3.14158, 3.14158]` and the pose in use sitting only
~0.0016 rad from one of those two limits, curobo's motion planning was suspected of being biased
toward long wrap-around solutions near that boundary. Rather than only re-tuning the numeric pose
(a band-aid that leaves the same limit boundary wherever it currently points), the arm's own mount
orientation on the chassis was rotated 180° about its own local Z axis, moving that wrap-around
boundary to point in the opposite chassis-relative direction.

The whole rotation is expressible as a single `quat`/`origin rpy` change, since the fixed
`chassis_link → default_mount → vx300s/base_link` chain collapsed into a single body during
URDF→MJCF conversion (those links have no joint of their own) — `vx300s/shoulder_link`'s own `quat`
in `miss.xml` (`waist`'s own body) and the equivalent `vx300s_joint`'s `origin rpy` in `miss.urdf`
(curobo's kinematics source). Every joint downstream (shoulder/elbow/forearm/wrist/gripper) is
defined relative to that same frame and comes along for free — no other file needed the change
(`curobo/miss.yml` and its collision spheres are link-relative, `get_grasp_pose_wrt_robot` operates
in `ee_gripper_link`'s own local frame, `robot_yaw`/`table_angle` in the task file are chassis/
world-frame conventions with no reference to the arm's mount at all — all confirmed by a dedicated
read-only investigation before touching anything).

**A real, self-caught mistake happened while implementing this**: the first attempt rotated the
MJCF body's `quat` by *pre*-multiplying a world-frame `Rz(180°)`, while the URDF's `origin rpy` was
computed by *post*-multiplying the same rotation in the joint's own local frame — two different
operations that only coincide if the rotation axis commutes with the body's existing orientation,
which it doesn't here. A direct MJCF-vs-curobo-FK consistency check (comparing `shoulder_link`'s
world pose from `mujoco.mj_forward` against `Miss.get_link_pose("vx300s/shoulder_link", ...)`)
caught the resulting mismatch immediately — the two files no longer agreed on the arm's orientation
for the same `qpos`. Recomputed correctly as a single local-frame (post-multiplied) rotation,
applied identically to both files, and re-verified: positions and orientations now match exactly
between MuJoCo and curobo.

**An incidental empirical finding from this debugging**: the `waist` joint's real rotation axis in
this hardware/mount, as authored, is **world X, not world Z** (confirmed by rotating the joint in
MuJoCo and reading off the incremental world-frame rotation matrix) — already true of the original,
untouched file, not something introduced by this change; the rotation preserves this same physical
axis exactly, only flipping which direction `waist=0` faces. Re-verified after the fix: joint range
unchanged (`[-3.14158, 3.14158]`), curobo's self-collision `IKSolver` still succeeds at multiple
sample configurations, and a `waist`×gimbal contact sweep found no new contacts needing the
existing `<exclude body1="world" body2="vx300s/shoulder_link"/>` to be revisited.

`ARM_PRESETS["home"/"upright"/"sleep"]` (still `waist=0.0`, unverified against the rotated mount —
only `"desk_reach"` has been re-checked) and their surrounding comments were updated to flag this
explicitly rather than silently leaving stale collision-free claims in place.

## Gripper-close timing: from a fixed schedule to a live proximity gate

Once motion planning and dataset recording both worked end-to-end, the user reported the gripper
closing "too early" — visibly closing while the arm was still approaching, then dragging the closed
gripper into the object. Traced to the executor's actual architecture, not a missing distance
check that could just be tightened: `MotionPlannerAgent.synthesize()`
(`src/simple/agents/mp.py`) pre-plans the **entire** episode's action queue *offline*, before any
physics runs, from `decompose()`'s fixed phase order (`OpenGripperSpec → GraspObjectSpec("approach")
→ CloseGripperSpec("grasp") → LiftSpec`); `CloseGripperSpec` queues 10 `close_eef` commands
immediately after the approach trajectory's own queued waypoints, with **no proximity check at
all** — closing is scheduled purely by position in the queue. `cli/datagen.py`'s
`execute_action_sequence` then dequeues exactly one action per `env.step()` (~17 physics substeps
at `physics_dt=0.002`/`render_hz=30`), with no wait for the P-only position actuators (no integral
term, a real, persistent steady-state droop under gravity/friction, already documented elsewhere in
this file) to actually catch up to each waypoint.

Two incremental fixes were tried and found insufficient before landing on the real one — kept here
because they clarify why the final design looks the way it does:
1. Compute FK on the *already-planned* trajectory (via the persistent, already-loaded
   `init_js_ik_solver` — one batched GPU call, no new planning) to find the first waypoint within
   some threshold of the grasp target, and start closing there instead of only at the very end.
   This only changes *where in the plan* closing starts, not whether the *real* robot has caught up
   to that point in real time — user-confirmed still too early even after shrinking the threshold
   from 5cm to 5mm.
2. Ramping the gripper's commanded qpos smoothly from open to closed across that same planned tail
   (instead of snapping straight to the closed target) fixed the *abruptness* but not the *timing*,
   for the same underlying reason.

**The actual fix**: a genuine closed-loop wait, gated on the robot's real, live position during
rollout rather than anything computed at planning time. The approach trajectory is now queued
untouched (gripper stays open throughout); right after it, a `"_grasp_close_gate"` marker
`ActionCmd` is queued carrying the true grasp target position, a distance threshold
(`close_within`, default `0.01`m), and a pre-built smooth open→closed ramp. `MotionPlannerAgent`
overrides `PrimitiveAgent.get_action()` to special-case this marker: on every call, while it sits
at the front of the queue, it reads the robot's **actual** live joint state from `observation
["agent"]` (confirmed to come straight from the simulator, not the plan), computes the real
end-effector position via `robot.fk()`, and — if still farther than `close_within` from the target
— returns a "hold at the final approach pose" action *without* advancing the queue, re-checking on
every subsequent step; only once genuinely within threshold does it pop the marker and let the
ramp play. Verified with an isolated simulation of the gate logic (no curobo/CUDA needed for this
part): correctly held through several steps of a simulated slow convergence and only released the
ramp once the *live* simulated distance actually crossed the threshold. Not yet confirmed by the
user against the real simulator at the time of writing.

## Known limitations / TODOs

- **The wheeled base is modeled in full (chassis, all 4 wheel joints, real meshes) but not
  driven.** `base_link` has no root joint (rigidly attached to the world, like every other
  manipulator robot in SIMPLE today); the wheel joints exist so the asset is genuinely complete,
  but are never actuated. No existing SIMPLE robot or task drives a wheeled base — building that
  (a real mobile/free joint on `base_link`, wheel actuation, and whatever engine/task-family
  support a driven base needs) is a separate, substantially larger effort, deliberately deferred.
- **`gimbal_cam`'s orientation was fixed by interactive tuning, not analytically derived.**
  `realsense2_description` (which would define the D455's real optical-frame transform) is still
  not vendored in this checkout, and the two guessed 180°-flip candidates tried earlier (both
  showed only the gimbal's own nearby mount hardware) were reverted. Instead,
  `scripts/miss_camera_tuning.py` was built for the user to tune `gimbal_cam`'s `pos`/`quat`
  interactively against the live MuJoCo viewer (`view cam` shows exactly what the camera sees,
  `nudge`/`tilt` adjust it live). The user-confirmed result —
  `pos="0.075398 0.086286 -0.024259" quat="0.579228 0.405580 -0.405580 -0.579228"` — is now the
  value in `miss.xml`, moved forward (toward +x in `gimbal_head`'s frame, past the mount bracket)
  and tilted down relative to the original placeholder, matching the earlier suspicion that the
  fix needed a position/standoff correction, not just a rotation.
- **`cube_2x2x2_v1_1.stl` (the D455 mount bracket) renders at ~2mm overall extent** with the same
  `scale="0.001 0.001 0.001"` that correctly sizes every other `misskal_description` mesh
  (`gimbal_base_`/`pescoco_v4_1`/`tela_iteracao_v5_1` all look right at that scale) — this one
  specific source STL appears to already be authored in meters, not millimeters like its
  siblings. Worth checking against the original CAD source.

## Task: `miss_tabletop_grasp_mp` (`simple/MissTabletopGraspMP-v0`)

Minimal validation task, modeled on `viperx_tabletop_grasp_mp.py`: Miss spawns next to a desk with
a single banana (`graspnet1b:5`) on it, no distractors. See
`src/simple/tasks/miss_tabletop_grasp_mp.py`'s own class docstring for the full detail; summary:

- **Four adjustable "knob" constructor parameters** describe the desk's placement relative to the
  robot: `table_distance` (m), `table_angle` (rad, direction from robot to desk), `table_height`
  (m), and `robot_yaw` (rad). `robot_yaw` — not a table `rotation_z` — is how "orientation of the
  robot to the table" is implemented: `TabletopSceneDRCfg.rotation_z` has **no effect** when
  `scene_manager="hssd"` (confirmed by reading `TabletopSceneDR.__call__`,
  `src/simple/dr/scene.py`: the hssd branch hardcodes `rotation_z = 0.0`), so the *robot's* own
  orientation is rotated instead via `SpatialDRCfg.robot_orientation_region` — equivalent for a
  single fixed robot + single table scene, since only the relative angle matters, and this
  mechanism is fully supported regardless of scene manager.
- **`robot_region` is an explicit 3-element `[x, y, z]` `Box`**, not the more common 2-element
  `[x, y]` form every other tabletop task uses. `SpatialDR` special-cases a 2-element
  `robot_region` by setting the robot's world Z to `-table_height` — correct for a robot bolted
  directly onto the tabletop (every other robot in SIMPLE), wrong for Miss, a wheeled platform
  that stands on the floor at Z=0 regardless of desk height. The 3-element form bypasses that.
- **The overview camera (`front_stereo`) needed very different distance/polar/azimuth values than
  any other task's.** `CameraEntity`'s spherical pose (`src/simple/core/actor.py`) always looks at
  world-frame `[0, 0, 0]`, which for a compact tabletop arm is right at the robot's own workspace
  height, so any reasonable values frame the arm+table well. Miss's origin is its floor-level
  chassis base — confirmed by rendering every combination near the other tasks' typical values
  (~1.5m distance, ~55-65° polar): the arm+desk (up around z=0.75-1.3m) never appeared or only
  clipped at the frame edge. `distance=2.5, polar=75°, azimuth=45°` was found by directly
  rendering several combinations until both the arm+gripper and the desk appeared in the same
  frame — not derived analytically, not claimed optimal. `gimbal_cam` (`mount="native"`) needs no
  such tuning (it moves with the gimbal correctly already) and, with its own pose now fixed (see
  "Known limitations" above), correctly shows the arm/gripper and the desk edge.
- **The robot was spawning with its wheels sunk ~6.35cm into the floor.** Robot Z was `0.0`, but
  `miss.xml`'s wheel geoms sit at body-frame `z=0.0345` with radius `0.098` — wheel bottoms at
  `-0.0635`, confirmed by rendering (the wheels visibly overlapped the checkered floor). Fixed
  with `_ROBOT_Z_OFFSET = 0.098 - 0.0345`, added to `robot_region`'s (3-element) Z value, so the
  wheel bottoms land exactly on world Z=0.
- **`table_distance` went through several rounds of correction** (chosen first from `gimbal_cam`'s
  field of view alone, `0.65`; then from chassis-vs-desk physical clearance using the wrong desk
  footprint, `1.4`; the current value, `1.75`, uses the *correct* desk footprint — see "The desk
  was never actually elevated" below for why the earlier physical-clearance number was itself
  based on the wrong actor). `table_height=0.70` and `robot_yaw=0.0` remain real-world-reasonable,
  untuned starting points. All four still need the established reachability diagnostic (watch
  `datagen`'s IK-fail rate) before this is genuinely usable, and `gimbal_cam`'s field-of-view
  framing needs re-verification from scratch against the corrected scene (established for the
  old, wrong desk placement, at completely different distances/heights).
- **The desk was never actually elevated in world coordinates at all — the most serious placement
  bug found in this integration.** `task.layout.scene.table.pose.position` (the DR-level,
  abstract pose) always correctly reflected `table_height`. But the *actually compiled* MuJoCo
  body for the `"table"` actor did not: `MujocoEngine._build_primitive`
  (`src/simple/engines/mujoco.py`) hardcodes, for any actor named exactly `"table"`,
  `table_position[2] = -0.5 * actor.size[2]` — unconditionally overriding whatever Z the DR
  computed. Confirmed directly: with `table_height=0.9` requested, `task.layout.scene.table.pose
  .position` said `z=0.85`, but the compiled body's actual `d.xpos` was `[.., -0.05]`. This is not
  a bug in isolation — every *other* tabletop task (Franka/Aloha/ViperX/WidowX AI) relies on it:
  those robots are bolted to the tabletop, so their own `robot_region`'s 2-element `[x, y]` form
  shifts the *robot* down by `-table_height` (`SpatialDR`, `src/simple/dr/spatial.py`) so that,
  relative to the robot's own shifted-down origin, the always-floor-level `"table"` appears to be
  at the right height. Miss can't use that trick (its base must sit at world Z=0, wheels on the
  floor — exactly why the 3-element `robot_region` form was used, see below), so with the robot no
  longer shifted down and the table hardcoded to floor level regardless of any DR value, **both
  ended up on the same plane** — exactly the "robot and desk spawning on the same plane" the user
  reported, and not something either earlier `table_distance` correction round could have fixed
  (both only ever addressed horizontal clearance). Fixed by switching the task to `table2`
  (`TabletopSceneDRCfg.enable_table2`/`table2_position`/`table2_height`) for the real, functional
  desk — `_build_primitive`'s Z-override only applies to the literal name `"table"`, and `table2`
  respects the DR-computed pose directly (confirmed: compiled `xpos` matched
  `task.layout.scene.table2.pose.position` exactly). The banana's surface-height lookup was
  rerouted to `table2` via `SpatialDRCfg.obj_surface_map = {"target": "table2"}`. The old `"table"`
  actor still exists (the hssd scene-building path creates it unconditionally) but is now inert
  floor-level set-dressing, kept centered under `table2` so its footprint doesn't sit exposed
  elsewhere in the room. `table2`'s own footprint (`scene.conf["table2_size"][:2]`, `[2.0, 0.7]`
  for `hssd:scene1`) is also *different* from `"table"`'s (`[1.27, 0.7112]`) — the physical
  chassis-clearance math had to be redone with the correct, larger footprint, which is why
  `table_distance`'s default moved again, from `1.4` to `1.75`.
- **Registry/env-construction bugs found and fixed along the way (core framework, not
  Miss-specific)**, both only ever exposed because interactive scripts were the first workloads in
  this whole codebase to construct the same task/robot uid more than once per process:
  - `RegistryMixin.make()` (`src/simple/core/registry.py`) used to memoize the first-built
    instance per uid and silently return that same cached object on every later call, ignoring any
    new constructor arguments entirely — confirmed directly (a second
    `MissTabletopGraspTaskMP(table_height=0.9)` silently returned the *first* instance, with
    `table_height` unchanged). Fixed by removing the memoization; `make()` now always constructs
    fresh, matching what every existing call site already assumed.
  - `BaseDualSim.__init__` (`src/simple/envs/base_dual_env.py`) constructed the task *twice*
    (once before building `self.mujoco`, again immediately after, reassigning `self.task`) — only
    ever harmless because both calls used to return the same memoized instance. Once the registry
    fix above landed, this produced two genuinely different Task objects (the engine holding a
    never-`reset()` one), raising `"call reset() first"` inside `env.reset()`. Fixed by removing
    the redundant second construction.
  - The cached-robot-object reuse across separate `gym.make()`/`env.close()` cycles (while the
    underlying MjModel/MjData were torn down and rebuilt fresh each time) is also a plausible
    direct cause of a segfault observed during this investigation, from robot state (e.g. curobo's
    lazily-cached kinematics/IK solver) outliving the MuJoCo buffers it was built against — not
    independently reproduced after the fix, but consistent with the fix's own mechanism.
- `get_grasp_pose_wrt_robot`'s identity-transform caveat (above) applies directly here — first-pass
  grasp attempts may be geometrically wrong even once IK/collision succeed.

## Camera tuning script

`scripts/miss_camera_tuning.py` -- the tool used to find `gimbal_cam`'s corrected pose above.
Same interactive terminal loop as `miss_interactive_control.py`, applied to a camera instead of
joints: `pos`/`quat`/`rpy` set the camera's pose absolutely, `nudge`/`tilt` adjust it relatively,
`view cam`/`view free` switch the MuJoCo viewer itself between showing exactly what the camera
sees and a third-person context view. Deliberately does not use a separate OpenCV window --
`cv2.namedWindow` raised "function is not implemented" on the user's own `opencv-python` build
(no GTK/Qt backend, common on nix-packaged Python), a system-dependency problem rather than a
SIMPLE one, so the already-working MuJoCo viewer's own multi-camera support
(`viewer.cam.type = mjCAMERA_FIXED`, `viewer.cam.fixedcamid`) is used instead.

Verified so far: `gym.make("simple/MissTabletopGraspMP-v0", ...)` → `env.reset()` succeeds, robot
spawns at the `desk_reach` pose (now on the rotated arm mount), the banana loads at the expected
height on `table2`, both cameras render correctly framed, and the groundplane sits exactly at the
wheels. A real `datagen` run (motion planning through an actual grasp attempt) now completes end to
end — motion planning reaches `MotionGenStatus` success (no more `IK_FAIL`), and a full episode's
frames (RGB images + the 6-dim arm-only action) save to a LeRobot dataset with no shape errors.
**Not yet confirmed**: whether the live gripper-close gate actually produces a natural-looking
close in the user's own manual test (the most recent change, described above), and whether the
grasp itself is geometrically correct once the gripper actually closes around the object — the
`get_grasp_pose_wrt_robot` rotation caveat (identity transform, never re-derived against
GraspNet's convention) means a successfully-planned, collision-free, well-timed grasp can still
close in the wrong orientation relative to the object; this remains open future work.
