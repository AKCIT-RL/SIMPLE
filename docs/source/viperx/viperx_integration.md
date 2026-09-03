# ViperX Integration

This document describes the integration of a single-arm Trossen ViperX manipulator into SIMPLE as
a new embodiment, on branch `feat/add-widowx`. Unlike the WidowX AI integration
(`docs/source/widowx/widowx_ai_integration.md`), which authored an entirely new robot from
scratch, ViperX is **extracted from Aloha's own left arm**: Aloha
(`src/simple/robots/aloha.py`) is two independently-rooted ViperX arms sharing one `Robot` class,
and the two arms turned out to already be structurally independent in the MJCF, URDF, and curobo
collision configuration — only the `Aloha` class itself (its `DualArm` protocol, `active_arm`
switching, `apply_action` arm-parking logic) couples them together. The task exposed as
`simple/ViperXTabletopGraspMP-v0` is the reference integration target.

## Assets

No new submodule or third-party source was needed. ViperX's assets live under
`data/robots/viperx/`, derived directly from Aloha's own `data/robots/aloha/` assets rather than
authored fresh:

- `viperx.xml` / `joint_position_actuators.xml` — Aloha's `aloha.xml` with the entire
  `right_base_link` body subtree, its actuator block, its `<contact><exclude>` entry, and its
  finger `<equality>` constraint deleted. The remaining `left_*` body tree, joint names, and
  actuator names are byte-identical to Aloha's own — confirmed to load standalone
  (`mujoco.MjModel.from_xml_path` → `mj_forward`, `nq=8, nu=8`) and to be genuinely collision-free
  at Aloha's own left-arm rest pose (`d.ncon == 0`).
- `viperx.urdf` — Aloha's `aloha.urdf` truncated to the shared `base_link` plus the `left_*`
  subtree (the URDF's `right_base_link` subtree runs from a single fixed `right_base_joint` to
  end-of-file with no shared content after it, confirmed by inspection, so this was a straight
  line-range truncation, not a hand edit).
- `assets/` — only the 13 mesh/texture files actually referenced by the left arm (`vx300s_1_base`
  through `vx300s_8_custom_finger_{left,right}`, `d405_solid.stl`, `interbotix_black.png`), copied
  from `data/robots/aloha/assets/` rather than referenced by relative path, so ViperX is a
  self-contained, independently distributable per-robot directory like every other robot in
  `data/robots/`.
- `curobo/viperx.yml` / `curobo/spheres/viperx.yml` — Aloha's `curobo/aloha.yml` /
  `spheres/aloha.yml` with every `right_*` entry removed (`link_names`, `collision_link_names`,
  `self_collision_ignore`, `self_collision_buffer`, `cspace.joint_names`/`retract_config`/weights)
  and `ee_link` fixed to `left_gripper_site` (no runtime arm-switching needed, unlike Aloha). The
  per-link sphere geometry values themselves are untouched — reused verbatim from Aloha's own,
  already-tuned fit.

Only `sim_mode="mujoco"` is supported (see Robot definition below); no single-arm USD was
authored.

## Robot definition (`src/simple/robots/viperx.py`)

`ViperX` is registered under uid `viperx` via `RobotRegistry`, composing `CuRoboMixin`,
`WristCamMountable`, and `HasParallelGripper` on top of the `Robot` base class — modeled directly
on `WidowXAI`'s structure (a genuine single-arm robot), not on `Aloha`.

**Degrees of freedom and joint mapping.** The robot has 6 arm joints
(`left_waist`/`left_shoulder`/`left_elbow`/`left_forearm_roll`/`left_wrist_angle`/
`left_wrist_rotate`) plus 2 independently-actuated finger joints
(`left_left_finger`/`left_right_finger`), for `dof=8`. Unlike WidowX AI's gripper (a single driven
joint plus a `<mimic>`-constrained mirror, `dof=7`), Aloha's gripper has no mimic joint — both
finger joints are real, separately-actuated DOFs — so both are counted and both are exposed to
curobo's `cspace.joint_names`.

**Controller.** `controller_cfg` is a `SingleArmBinaryEEFControllerCfg` with a `PDJointPosController`
for the 6 arm joints and a **generic** `ParallelGripperEEFController` for the two finger joints —
not a robot-specific subclass. This is a direct consequence of extracting Aloha's own left arm
rather than integrating new gripper hardware: Aloha's own `left_eef` config
(`src/simple/robots/aloha.py`) already uses this same generic controller with the same joint
names, so its hardcoded open/close ctrl values (`0.0205`/`0.0`,
`src/simple/robots/controllers/eef.py`) apply to ViperX unchanged. (WidowX AI needed a
`WidowXParallelGripperEEFController` subclass because its gripper travel, `0`–`0.044`, didn't
match those hardcoded values — ViperX's gripper is the identical hardware Aloha already uses, so
no such mismatch exists here.)

**Grasp-pose frame conversion (`get_grasp_pose_wrt_robot`).** Reused verbatim from
`Aloha.get_grasp_pose_wrt_robot`, including its non-identity `T_grasp_ee` flip matrix. This is a
property of the `*_gripper_site` frame convention itself (both of Aloha's gripper sites share the
same local convention), not of which arm a site belongs to, so extracting the left arm alone
changes nothing about the derivation. `robot_eef_offset = 0.00` is likewise reused unchanged from
Aloha's own value — Aloha's own comment marks it "HACK to avoid collision" rather than a value
derived from link geometry, and this was not re-derived for ViperX.

**Init pose.** Aloha's own left-arm rest pose (`left_waist=0, left_shoulder=-0.96, left_elbow=1.16,
left_forearm_roll=0, left_wrist_angle=-0.3, left_wrist_rotate=0`, gripper open at `0.04`) is reused
directly as both `controller_cfg`'s `init_qpos` and `init_joint_states` — no WidowX-AI-style
"class default pose is broken" override was needed. This was independently verified for the pruned
single-arm MJCF (`mujoco.mj_forward` → `d.ncon == 0` at this exact qpos: removing the right arm can
only remove self-collision pairs, never add any) and for curobo's own self-collision check (an
`IKSolver.solve_single` query seeded and targeted at this exact retract configuration succeeds with
`self_collision_check=True` — unlike WidowX AI, whose analogous trivial query fails at its own
class-level default pose).

## Task definition (`src/simple/tasks/viperx_tabletop_grasp_mp.py`)

Registered as `viperx_tabletop_grasp_mp` and exposed as the gym environment
`simple/ViperXTabletopGraspMP-v0` (`src/simple/envs/__init__.py`), reusing the generic
`TabletopGraspEnv`. Structurally mirrors `widowx_ai_tabletop_grasp_mp.py` (a genuinely single-arm
task) rather than `aloha_tabletop_grasp_mp.py`, whose `decompose()` and camera set assume a
bimanual robot.

**`decompose()`** omits the `hand_uid` kwarg entirely from
`OpenGripperSpec`/`GraspObjectSpec`/`CloseGripperSpec`/`LiftSpec`, instead of pinning it to
`hand_uid="left"` the way `aloha_tabletop_grasp_mp.py` does. `hand_uid` was already a plain
optional dict key in `subtask_spec.py`/`agents/mp.py` with no dual-arm assumptions baked in, so
simply not passing it was sufficient — no changes were needed to either of those files.

**Sensor and spatial-DR configuration** are reused verbatim from `aloha_tabletop_grasp_mp.py`
(the `front_stereo` camera and the left arm's own `wrist` eye-in-hand camera, and the
`robot_region`/`target_region`/`distractors_region` boxes), not re-derived. Since the pruned
`viperx.xml` keeps `left_base_link` at the exact same offset from the world/curobo `base_link`
frame as Aloha's own left arm (`pos="-0.469 -0.019 0.0"`, untouched), the same table/robot/target
geometry that was already tuned for Aloha's left arm applies unchanged.

## curobo motion planning

**Self-collision.** Aloha's own collision-sphere fit (`data/robots/aloha/curobo/spheres/aloha.yml`)
was reused verbatim for the left arm's spheres — no re-generation, re-fitting, or empirical
buffer/ignore tuning was needed, in contrast to WidowX AI, whose sphere fit required a
from-scratch analytic re-derivation against the MJCF's own collision primitives plus per-link
`self_collision_buffer`/`self_collision_ignore` tuning to resolve false-positive self-collisions.
This was verified directly: a trivial curobo `IKSolver` query (seed = target = the class-level
rest configuration) succeeds with `self_collision_check=True` on the pruned single-arm config,
which is exactly the query that fails for WidowX AI's analogous default pose.

**World collision — table.** `create_collision_world_cfg`
(`src/simple/mp/curobo.py`) special-cases Franka and WidowX AI
(`uid.startswith(("franka", "widowx_ai"))`) to skip the table as a world-collision obstacle
entirely, because those robots' `base_link` sphere protrudes measurably below the table's
registered top surface at `table_height=0`. **ViperX does not need this hack.** Aloha (never in
this list) already runs successfully in production with the same `base_link` geometry and the
same table mount height, and an end-to-end `datagen` run against
`simple/ViperXTabletopGraspMP-v0` (below) completed with no table-related `IK_FAIL` — so the
`uid.startswith(...)` tuple was left unchanged.

**Target-object collision.** As with every other task in this family, grasp-approach planning
requires `--ignore-target-collision` (`CuRoboPlanner`'s `ignore_target_collisions`, default
`False`) — a usage requirement of the task family, not a ViperX-specific issue.

## Dataset recording (`src/simple/envs/lerobot.py`)

The LeRobot dataset schema for the `"action"` feature is derived once from `task.action_space`,
which for `SingleArmBinaryEEFController` is deliberately arm-only (`{"arm": ...}`; the gripper has
its own, separate `eef_action_space`, never merged in) — 6-wide for ViperX. The per-step recorded
action vector is assembled separately by iterating every joint in
`action.parameters["target_qpos"]`, and a **generic** fallback path deduplicates `"*finger*"`-named
joints by side-prefix (`jname.split('_')[0]`) before appending — this path already exists to
support Aloha, and both of ViperX's finger joints (`left_left_finger`, `left_right_finger`) share
the same side-prefix (`"left"`), so it correctly collapses them to a single contributed value.

That single deduped gripper value is exactly the bug: for **Aloha**, this fallback's contribution
to `frame["action"]` is silently discarded and replaced wholesale by a separate `isinstance(self.robot, DualArm)`-gated
block further down in the same function (the `# FOR ALOHA` branch) — so the generic dedup path's
output for Aloha never actually reaches the recorded dataset. **ViperX is not a `DualArm`**, so no
such override exists, and the deduped single gripper value fell straight through into the recorded
vector — 7 values (6 arm + 1 gripper) against the declared 6-wide (arm-only) schema. This is the
same class of bug documented for WidowX AI's `left_carriage_joint` (a 7-vs-6 shape mismatch
rejected by `lerobot`'s `validate_frame`), reproduced empirically via one `datagen` run
(`ValueError: The feature 'action' of shape '(7,)' does not have the expected shape '(6,)'`) and
fixed the same way: an explicit skip condition, `"viperx" in self.robot.uid and "finger" in jname:
continue`, added before the generic dedup branch. Re-running `datagen` after the fix produced a
valid 153-frame episode with the expected 6-wide `action` feature.

## Collision-sphere renders

`scripts/render_collision_spheres.py` (previously Franka/WidowX-AI-only) was extended with a
`"viperx"` entry. Two curobo-collision links needed special handling beyond the existing
`body_map` mechanism: `left_gripper_camera` and both `left_custom_finger_{left,right}_link`
entries in `spheres/viperx.yml` are separate URDF-only links (for curobo's own kinematics) with
**no matching MJCF body** — the MJCF draws their meshes directly inside `left_gripper_base` /
`left_{left,right}_finger_link` — and, unlike WidowX AI's zero-offset `gripper_left`/
`gripper_right` case, their fixed-joint offset from the MJCF parent body is genuinely nonzero
(confirmed directly in `viperx.urdf`). All three offsets happen to be pure rotations about local
X, so `body_map` entries for these three links are `(parent_body_name, translation,
rotation_about_x_radians)` tuples instead of plain strings, and `build_spec` applies
`center' = Rx(θ) @ center + translation` to each sphere's center before parenting it — verified
visually below (the fingertip and wrist-camera spheres land exactly on the corresponding mesh
features, not offset from them).

![ViperX collision-sphere approximation, iso view, rest pose](media/spheres_rest_iso.png)

*Rest pose (Aloha's own left-arm rest qpos), isolated sphere approximation. Each color is one
link. The near-invisible sphere at the shoulder reflects `spheres/viperx.yml`'s own
`left_shoulder_link` radius of `0.0001` (inherited unchanged from Aloha) rather than a rendering
issue.*

![ViperX collision-sphere approximation, iso view, grasp pose](media/spheres_grasp_iso.png)

*A genuine arm configuration reached near the end of a real `datagen` episode's close+lift phase
(not hand-picked), gripper closed.*

| Real mesh (front view, grasp pose) | Sphere approximation (same pose) |
|---|---|
| ![ViperX mesh render, grasp pose, front view](media/mesh_grasp_front.png) | ![ViperX sphere approximation, grasp pose, front view](media/spheres_grasp_front.png) |

*Side-by-side comparison at the same pose and camera. The two fingertip spheres (red/blue) and the
tiny wrist-camera sphere (magenta) land on the corresponding real mesh features — the
offset-transform fix described above, not a coincidence of the render script's default zero-offset
mapping.*

## Verification performed

- Standalone MJCF load (`mujoco.MjModel.from_xml_path` → `mj_forward`): `nq=8`, `nu=8`, and
  `d.ncon == 0` at Aloha's own left-arm rest pose.
- curobo config load (`CuRoboMixin.__init__` via `ViperX()`): resolves `robots/viperx/curobo/viperx.yml`
  without error; `ee_link == "left_gripper_site"`; `cspace.joint_names` is the expected 8 left-arm
  joints.
- curobo self-collision: a trivial `IKSolver.solve_single` query (seed = target = rest qpos)
  succeeds with `self_collision_check=True` (and with it `False`) — no broken-default-pose issue
  of the kind found for WidowX AI.
- End-to-end `datagen` (`simple/ViperXTabletopGraspMP-v0`, `--sim-mode mujoco --headless
  --ignore-target-collision`): completed successfully after the `lerobot.py` fix above, writing a
  153-frame episode with a 6-wide `action` feature (verified by reading the written parquet
  directly) and no table-related `IK_FAIL`.
- Collision-sphere renders (`scripts/render_collision_spheres.py --robot viperx`) at `rest` and
  `grasp` poses, visually cross-checked against the real mesh at matching camera poses.
