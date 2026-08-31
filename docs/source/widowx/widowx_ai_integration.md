# WidowX AI Integration

This document describes the integration of the Trossen WidowX AI manipulator into SIMPLE as a new
embodiment, on branch `feat/add-widowx`. It covers the asset/submodule setup, the robot and task
definitions, the shared-engine changes required to support the new MJCF, the curobo motion-planning
configuration (self-collision, world-collision, grasp-pose conversion), and the dataset-recording
fixes required for `datagen` to produce valid episodes. The task exposed as
`simple/WidowXAITabletopGraspMP-v0` is the reference integration target.

## Assets and submodules

Two submodules were added under `third_party/`, following the existing pattern used for other
robot/teleop integrations:

- `third_party/trossen_arm_description`: URDF (`wxai_follower.urdf`) and mesh sources.
- `third_party/trossen_arm_mujoco`: MJCF (`wxai_follower.xml`) and a reference diff-IK controller,
  registered as an editable `uv` dependency (`trossen-arm-mujoco`) under a new `trossen` optional
  group in `pyproject.toml`.

Robot assets consumed by SIMPLE at runtime (MJCF, meshes, curobo config, collision-sphere
description) live under `data/robots/widowx_ai/`, resolved via `resolve_data_path()` per the
project's lazy-download convention; this directory is not committed (`data/` is repository-ignored)
and is intended for distribution through the `USC-PSI-Lab/SIMPLE` HuggingFace dataset.

No USD asset exists for WidowX AI. `WidowXAI.usd_path` is set to a placeholder path for API
symmetry with other robots, but does not resolve to a real asset; only `sim_mode="mujoco"` is
currently supported.

## Robot definition (`src/simple/robots/widowx_ai.py`)

`WidowXAI` is registered under uid `widowx_ai` via `RobotRegistry`, composing `CuRoboMixin`,
`WristCamMountable`, and `HasParallelGripper` on top of the `Robot` base class, consistent with the
existing Franka/Aloha/Vega robot definitions.

**Degrees of freedom and joint mapping.** The robot has 6 arm joints (`joint_0`..`joint_5`) plus a
single actuated gripper joint, `left_carriage_joint`. `right_carriage_joint` mirrors
`left_carriage_joint` through a MuJoCo `<equality>` mimic constraint (`polycoef="0 1 0 0 0"`) in the
MJCF, and is deliberately excluded from `WidowXAI.joint_names`/`dof` (7, not 8): both curobo's
`cspace.joint_names` (in `curobo/widowx_ai.yml`) and `CuRoboMixin.fk()`/`get_robot_qpos()` require a
1:1 correspondence with this list, and treating a driven mimic joint as an independent DOF would
break that correspondence. `setup_control()` additionally sets `right_carriage_joint`'s qpos
directly from `left_carriage_joint`'s value at controller-setup time, because MuJoCo only resolves
`<equality>` constraints through `mj_step`'s constraint solver, not through `mj_forward` — without
this, the two gripper fingers can visibly desynchronize for one or more steps after `reset()`.

**Gripper controller.** WidowX AI's gripper travel (`0`–`0.044`, from the MJCF joint range) does not
match Franka's hardcoded `0.0`/`0.0205` open/close values in the shared
`ParallelGripperEEFController`. Rather than parameterizing that shared class, a robot-specific
subclass, `WidowXParallelGripperEEFController`, overrides `open_gripper`/`close_gripper` with
`OPEN_CTRL = 0.044` / `CLOSE_CTRL = 0.0`.

**Grasp-pose frame conversion (`get_grasp_pose_wrt_robot`).** This method converts a GraspNet grasp
pose into the pose curobo's own `ee_link` expects. For Franka, curobo's `ee_link` is `fr3_hand` — a
wrist frame distinct from the gripper contact point (`fr3_hand_tcp`) — so the conversion combines a
translation (TCP → wrist) with a rotation remap (GraspNet's approach axis, always local +X by
convention, to `fr3_hand`'s own approach axis, local +Z). For WidowX AI, curobo's `ee_link` is
`ee_gripper_link`, which *is* the gripper contact point: the URDF's fixed joint from `link_6` to
`ee_gripper_link` is a pure translation (`<origin xyz="0.156062 0 0"/>`, no rotation), so
`ee_gripper_link` shares `link_6`'s orientation exactly. `link_6`'s own approach axis is local +X
(the direction to the gripper) and its finger-closing axis is local +Y (from the MJCF gripper joint
axes, `left_carriage_joint axis="0 1 0"`, `right_carriage_joint axis="0 -1 0"`) — matching GraspNet's
(approach=X, binormal/closing=Y) convention exactly. Both transform matrices (`T_ee_hand`,
`T_grasp_ee`) are therefore identity for WidowX AI, rather than reused from the Franka
implementation. This was verified numerically: a canonical straight-down GraspNet grasp, passed
through the identity transform and curobo forward kinematics, places `ee_gripper_link`'s own +X axis
at world `[0, 0, -1]`.

## Task definition (`src/simple/tasks/widowx_ai_tabletop_grasp_mp.py`)

Registered as `widowx_ai_tabletop_grasp_mp` and exposed as the gym environment
`simple/WidowXAITabletopGraspMP-v0` (`src/simple/envs/__init__.py`), reusing the generic
`TabletopGraspEnv`.

**Object placement region.** `_AnnularSectorBox` (a `Box` subclass) samples target/distractor
positions from an angular sector of an annulus around the robot base, rather than an axis-aligned
rectangle, to keep objects within WidowX AI's reach envelope while facing the direction the robot is
mounted toward the table. It satisfies the informal interface `SpatialDR._random_place_one_object`
actually calls (`sample()`, `.low`/`.high` for episode-metadata logging) without being a true `Box`.

**Initial pose override.** `WidowXAI`'s class-level default pose is all-zero joints, which curobo's
default self-collision configuration (see below) incorrectly flags as colliding. The task instead
spawns the robot at a "close-high" joint configuration
(`_CLOSE_HIGH_ARM_QPOS = [0.0, 0.1363, 0.8843, -0.7480, 0.0, 0.0]`, gripper open), solved once via
curobo IK with self-collision checking disabled and independently cross-checked as collision-free
against MuJoCo's own collision primitives (`mj_forward` → `ncon == 0`) at the same qpos. This
requires overriding two independent attributes on the robot instance, not one:

- `controller_cfg` — determines the physical reset qpos MuJoCo applies
  (`PDJointPosController.set_initial_qpos`).
- `init_joint_states` — a separate dict consumed by `CuRoboMixin.planning_init_joint_states()` as
  the seed configuration for the first planned trajectory (`CuRoboPlanner.batch_plan_for_approach`),
  and read elsewhere as the robot's generic "nominal" joint state.

Overriding only `controller_cfg` produces a robot that visually spawns at close-high but whose first
planned trajectory is computed as if starting from all-zero — observed directly as a joint-space
"snap" from close-high toward all-zero over the first handful of simulation steps, with the recorded
action for those steps at approximately zero (i.e., not a commanded motion). Both attributes are now
set together, as instance attributes on the task-owned robot object, leaving the `WidowXAI` class
defaults untouched.

## Shared-engine changes (`src/simple/engines/mujoco.py`)

Three changes were required in the shared MuJoCo engine to load and render WidowX AI's MJCF; all
three are backward-compatible with existing robots.

1. **Ground-plane material.** The engine's ground-plane geom referenced a `"groundplane"` material
   that SIMPLE itself has never defined — every previously integrated robot MJCF happened to bundle
   its own `"groundplane"` texture/material as a side effect of demo-authoring, so the dependency was
   implicit. WidowX AI's MJCF does not define one, causing a compile error. The material is now
   defined explicitly and unconditionally (guarded by `mjSpec.material("groundplane") is None`, to
   avoid a duplicate-definition error for robots that already provide one), using the same texture
   parameters Franka's MJCF happened to provide, making this a visual no-op for existing robots.
2. **Native camera mounts.** Added a `camera.mount == "native"` branch for cameras that are already
   defined and posed inside the robot's own MJCF (WidowX AI's wrist camera, `"cam"`, in
   `wxai_follower.xml`), instead of constructing a duplicate camera element under a different mount
   convention. The branch asserts the named camera already exists in the loaded model; the render
   pipeline picks it up by name via `sensor_cfgs`, unchanged.
3. **Renderer lookup guard.** The camera-rendering loop iterated all MJCF cameras and looked up a
   renderer for each by name, which raised `KeyError` for any MJCF camera not declared in the task's
   `sensor_cfgs` (e.g., WidowX AI's `"cam"` when not used as a sensor, or Aloha's
   `"teleoperator_pov"`). Changed to `self.renderers.get(...)` with an explicit skip on `None`.

## curobo motion planning

### Self-collision spheres

Collision spheres for curobo (`data/robots/widowx_ai/spheres.yml`) are generated by
`data/robots/widowx_ai/generate_spheres.py` against the MJCF's own simplified collision primitives
(cylinders, capsules, boxes), not the detailed visual STL meshes. This distinction matters: several
links (e.g. `base_link`) have collision geometry in the MJCF that is measurably coarser (larger
radius) than their visual mesh, and this coarser geometry — not the visual mesh — is what MuJoCo
actually simulates contacts against. Fitting spheres to the visual mesh instead would in some cases
produce spheres *smaller* than the real simulated collision volume, trading a false-positive
collision problem for a false-negative one (curobo approving poses that collide for real). Sphere
generation therefore reads geometry primitives and dimensions directly from the MJCF
(`_analytic_spheres_for_geom` for single primitives, `_merged_obb_spheres_for_link` for
multi-geom links), rather than performing mesh-based fitting for links that have MJCF collision
primitives available.

Two link-level configuration surfaces, both in `data/robots/widowx_ai/curobo/widowx_ai.yml`, adjust
the effective collision margin without changing the sphere radii recorded in `spheres.yml`:

- `self_collision_buffer` (per-link): a positive value *increases* required clearance (stricter);
  a negative value *decreases* it (looser). Used for `base_link` (`-0.003`) and `link_4`
  (`-0.015`), both tuned empirically and cross-checked against `mujoco.mj_forward`/`ncon` on the
  same poses.
- `self_collision_ignore` (per link, listing other links to exclude from checking against):
  extended for `link_6` to include `gripper_left`/`gripper_right`, in addition to the existing
  `carriage_left`/`carriage_right` exclusion. This is a link-pair exclusion of last resort, applied
  only after `self_collision_buffer` was confirmed to have no effect on this pair (verified
  directly) and no MJCF collision geometry exists for the gripper pads to validate against.

**Sphere-approximation renders.** `scripts/render_collision_spheres.py` renders the generated sphere
approximation (one color per link) alongside the robot's own visual mesh, for direct visual
comparison, at named reference poses (`rest`, `close-high`).

![WidowX AI collision-sphere approximation, iso view, rest pose](media/spheres_rest_iso.png)

*Rest pose, isolated sphere approximation. Each color is one link (deterministic per-link hue,
independent across renders). The elongated chains along the two forearm links are visibly capped by
spheres that bulge slightly past each link's own flat end faces — an unavoidable artifact of
approximating a flat-ended cylindrical/box segment with a chain of spheres, since a sphere has no way
to represent a flat face.*

![WidowX AI collision-sphere approximation, iso view, close-high pose](media/spheres_close-high_iso.png)

*The same approximation at the "close-high" spawn pose used by the task. The base-link cluster
(top-right in this view) sits close enough to the folded forearm chain that the flat-cap bulge
described above becomes the dominant contributor to the residual `base_link`↔`link_2` self-collision
margin discussed above.*

| Real mesh (side view, close-high) | Sphere approximation (same pose) |
|---|---|
| ![WidowX AI mesh render, close-high, side view](media/mesh_close-high_side.png) | ![WidowX AI sphere approximation, close-high, side view](media/spheres_close-high_side.png) |

*Side-by-side comparison at the same pose and camera. The sphere approximation visibly grows past
the true silhouette at the base and end-effector clusters — geometry inflation that is expected and
budgeted for by the self-collision buffer/ignore settings above, not a rendering defect.*

### World collision — table

`create_collision_world_cfg` (`src/simple/mp/curobo.py`) already special-cased Franka to skip the
table as a world-collision obstacle entirely (a pre-existing comment: "HACK franka can not has
table"). WidowX AI hits the same underlying issue: `base_link` is mounted flush to the table
(`table_height=0`), and its own bottom-face sphere — even after the self-collision fixes above —
protrudes measurably (~1.3 cm, confirmed directly) below the table's registered top surface, which
curobo reports as a permanent `IK_FAIL` from the start state alone.

curobo exposes no mechanism to exclude a single robot-link/world-obstacle pair (only
`self_collision_ignore`, which is robot-versus-robot only; `disable_link_spheres` is a global toggle
that would also remove the link from self-collision checking). The alternative considered was
shrinking the table's collision geometry by a small margin; the table-skip approach already accepted
for Franka was reused for WidowX AI instead, extending the uid check from `startswith("franka")` to
`startswith(("franka", "widowx_ai"))`.

### Target-object collision

Independently of the table fix, grasp-approach planning still failed with the target object itself
registered as a collision obstacle. `CuRoboPlanner`'s `ignore_target_collisions` parameter (exposed
as the `--ignore-target-collision` `datagen` CLI flag) defaults to `False` and must be explicitly
enabled for grasp-approach planning to succeed — this applies generally, not only to WidowX AI, and
is a usage requirement for this task family rather than a code defect.

## Dataset recording (`src/simple/envs/lerobot.py`)

The LeRobot dataset schema for the `"action"` feature is derived once from
`task.action_space`, which for `SingleArmBinaryEEFController` is deliberately arm-only
(`{"arm": ...}`; the gripper has its own, separate `eef_action_space`, never merged in) — 6-wide for
WidowX AI. The per-step recorded action vector, however, is assembled separately by iterating every
joint in `action.parameters["target_qpos"]` and skipping gripper joints by hardcoded, per-robot name
matching (`"finger"` substring matches for Franka/Aloha). WidowX AI's gripper joint,
`left_carriage_joint`, matched none of the existing patterns, so it fell through into the recorded
vector, producing a 7-wide action against the declared 6-wide schema and causing every frame to be
rejected (`ValueError: ... shape '(7,)' does not have the expected shape '(6,)'`) — `datagen` ran to
completion with motion planning succeeding, but wrote no episode data. Fixed with an explicit skip
condition for `widowx_ai` + `left_carriage_joint`, following the same per-robot, name-based pattern
already in use, without generalizing the mechanism (which would also affect Aloha's own dedup logic
for its two finger joints).

## Known open issue

With the fixes above in place, `datagen` reaches the lift phase of the task and fails there with
`'IKResult' object has no attribute 'status'`. This is unrelated to the issues described above (a
different code path, `batch_plan_for_lift`/`batch_plan_for_lift_bodex` in
`src/simple/mp/curobo.py`) and had not previously been reached, since earlier planning phases were
failing first. Not yet investigated.


