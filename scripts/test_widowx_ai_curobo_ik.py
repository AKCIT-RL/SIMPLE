"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Waypoint-based validation that CuRobo actually controls the WidowX AI arm + gripper.

WHY THIS SCRIPT EXISTS
-----------------------
`docs/teleop_simple_study/widowx_ai_integration_status.md` (section 10) flags that end-to-end
motion-planning for the WidowX AI integration was never confirmed to work -- the only thing
validated was that `RobotConfig.from_dict(widowx_ai.yml)` loads and FK at qpos-zero looks
plausible. This script closes that gap with a *quantitative* pass/fail check instead of an
eyeballed one: it drives the real WidowX AI robot inside an actual SIMPLE MuJoCo env (same
`gym.make(...)` / `ActionCmd` / `apply_action` path `datagen` and `eval` use) through a short
list of Cartesian waypoints, solving each one with CuRobo, and reports the *measured* end
-effector position error (FK-after-settling vs. commanded target) instead of "it looked right
in the viewer".

It intentionally does NOT try to reproduce `third_party/trossen_arm_mujoco/.../wxai_follow_target.py`
(mouse-draggable mocap target). That script's IK is Trossen's own damped-least-squares
`Controller` class, not CuRobo, and SIMPLE has no mocap-body support in its MJCF scene assembly
(`engines/mujoco.py`) to begin with -- adding one would be new engine surface for a purely
cosmetic win. A scripted waypoint list gives the same "does CuRobo actually move the robot"
answer with a number attached, and reuses only code paths that already exist and are already
used by `datagen`/`keyboard_agent.py`.

HOW CUROBO CONTROLS THE WIDOWX AI, CONCRETELY
-----------------------------------------------
1. Config (`data/robots/widowx_ai/curobo/widowx_ai.yml`) declares the robot's kinematic chain to
   CuRobo: URDF path, `base_link`, `ee_link=ee_gripper_link`, and `cspace.joint_names` -- the 7
   joints CuRobo is allowed to move: `joint_0..5` (the 6 arm joints) + `left_carriage_joint` (the
   gripper's *position-controlled* travel; `right_carriage_joint` is a URDF `<mimic>` of it, not
   an independent DOF, so it is excluded -- see `widowx_ai.py`'s `dof=7`). This joint order is
   exposed at runtime as `robot.kin_model.joint_names`, and it is what every CuRobo call below
   is implicitly indexed by.
2. `CudaRobotModel` (built from that config, see `CuRoboMixin._get_kinematic_model` in
   `src/simple/robots/mixin.py`) does **forward kinematics**: given a 7-vector of joint angles it
   walks the URDF chain (via cuRobo's parsed kinematic tree, not MuJoCo) and returns the
   `ee_gripper_link` pose. This is `robot.fk(qpos)` -- used below only to *measure* how well IK
   did, not to command anything.
3. `IKSolver` (`curobo.wrap.reacher.ik_solver.IKSolver`, GPU-batched nonlinear optimizer, not a
   closed-form/Jacobian-transpose solver) does **inverse kinematics**: given a target
   (position, quaternion) it searches joint space for a 7-vector whose FK matches the target
   within tolerance, optionally biased toward a "retract"/seed configuration (here: the robot's
   *current* qpos, so the solver prefers the nearest solution instead of a random one -- avoids
   large, un-physical joint jumps between waypoints).
   `CuRoboMixin.ik(p, q, current_joint)` in `mixin.py` wraps exactly this, and is the same call
   `keyboard_agent.py`'s Cartesian mode and every `baselines/*.py` policy adapter uses to turn a
   predicted/desired EEF pose into joint targets. This script uses the same call.
4. The IK solution (a 7-vector matching `kin_model.joint_names` order) is packed into
   `target_qpos = dict(zip(robot.kin_model.joint_names, solution))` and sent as
   `ActionCmd("move_qpos_with_eef", target_qpos=target_qpos, eef_state="open_eef"|"close_eef")`.
   `WidowXAI.apply_action` (`robots/widowx_ai.py`) unpacks it and does
   `self.actuators[jname].ctrl = jval` for each of the 6 arm actuators -- MuJoCo's own PD/position
   actuators (defined in `wxai_follower.xml`) then close the loop against real physics over the
   next `env.step()` calls; CuRobo itself does not simulate dynamics, it only picks the setpoint.
   `eef_state` is handled separately by `WidowXParallelGripperEEFController.open_gripper/
   close_gripper`, which hardcodes the `left_carriage_joint` actuator to 0.044 (open) or 0.0
   (closed) -- CuRobo's own IK value for that joint is a byproduct of solving cspace, not what
   actually drives the gripper.
5. Collision awareness: `CuRoboMixin.ik()` builds its `IKSolver` via
   `_create_empty_world_ik_solver()` -- "empty world" means **no obstacle collision checking**
   (no table, no target object) -- only *self*-collision (robot-against-itself) is checked, via
   `data/robots/widowx_ai/spheres.yml`. See the CRITICAL FINDING below: on this machine that
   self-collision check is what was actually broken.

CRITICAL FINDING FROM BUILDING THIS SCRIPT (verified, not assumed)
---------------------------------------------------------------------
`robot.ik()` / `CuRoboMixin._create_empty_world_ik_solver()` (self-collision ON, the default)
fails for *every* WidowX AI target tried, including solving IK for the pose the robot is
*already at* (seeded with that exact retract config -- a trivially satisfiable query). Isolating
`self_collision_check=False` on an otherwise-identical `IKSolver` makes the exact same query
succeed. The same default-on self-collision check on the *Franka* FR3 (`robot_uid="franka_fr3"`)
does not fail. This was cross-checked one level up too: `simple.mp.curobo.CuRoboPlanner
.batch_plan_for_move` (the actual `MotionGen`-based planner `datagen` uses, not just the toy
`IKSolver` here) fails with `MotionGenStatus.IK_FAIL` on a target 3 cm above the robot's own
resting pose -- i.e. **this is very likely the root cause of the "systematic IK_FAIL" risk
already flagged in section 10 of the integration status doc**, and it is unrelated to the
`get_grasp_pose_wrt_robot` rotation-matrix concern raised in that same section.

The likely culprit is `data/robots/widowx_ai/spheres.yml` (auto-generated by
`data/robots/widowx_ai/generate_spheres.py` from the per-link STL meshes). Several `link_2`/
`link_3` collision spheres have local-frame centers as far as ~0.27 m from that link's origin
with a ~0.031 m radius -- physically implausible for WidowX AI's individual arm segments (the
whole 6-DOF chain reaches roughly 0.4-0.5 m), which reads as spheres fit against meshes that are
not correctly localized to their own link frame, causing adjacent links to be flagged as
overlapping/self-colliding at essentially every configuration, including the robot's own
neutral pose. This script does **not** attempt to fix `spheres.yml` -- that's a separate,
dedicated task (re-check `generate_spheres.py`'s mesh inputs/frames, or re-fit with a smaller
`surface_sphere_radius`, then re-validate against real MuJoCo self-collision geoms). Until then,
`--no-self-collision` (the default here) is required for *any* IK/motion-planning call on
`widowx_ai` to succeed -- pass `--self-collision` to reproduce the broken default and see it
fail.

WHAT THIS SCRIPT DOES NOT VALIDATE
-------------------------------------
- Collision avoidance against the table/scene/target object (`ik()` uses an empty world by
  design -- see point 5 above). A waypoint succeeding here says nothing about whether a full
  `datagen` grasp trajectory through clutter will.
- The grasp-pose rotation convention in `WidowXAI.get_grasp_pose_wrt_robot` (still flagged
  unverified in the status doc) -- this script commands raw Cartesian waypoints, not grasp poses.
- IsaacSim / USD (no USD exists for WidowX AI yet; `sim_mode` is hardcoded to a MuJoCo-only mode
  here for that reason).

USAGE
-----
Quantitative check only, fast, no GUI (recommended first run):
    .venv/bin/python scripts/test_widowx_ai_curobo_ik.py --headless

Watch it move in the MuJoCo viewer (slower, needs a display):
    .venv/bin/python scripts/test_widowx_ai_curobo_ik.py --no-headless

Step through waypoints by hand -- settles, prints the result, then holds the pose (physics keeps
running so the actuators keep fighting gravity, not a frozen frame) until you press Enter to move
to the next one. Only meaningful with `--no-headless` (nothing to look at otherwise), but not
enforced -- combine with `--headless` if you just want the manual pacing on stdout:
    .venv/bin/python scripts/test_widowx_ai_curobo_ik.py --no-headless --interactive

Reproduce the broken default (self-collision on) to see every waypoint fail IK:
    .venv/bin/python scripts/test_widowx_ai_curobo_ik.py --headless --self-collision

Tighter/looser convergence check, more/less settling time per waypoint:
    .venv/bin/python scripts/test_widowx_ai_curobo_ik.py --pos-tol 0.005 --settle-steps 120

Exit code is 0 iff every waypoint both (a) found an IK solution and (b) settled within
`--pos-tol` of the commanded position. `--interactive` does not affect this -- Ctrl+C during a
wait aborts the whole run instead of counting remaining waypoints as failed.
"""

from __future__ import annotations

import sys
import threading
from typing import Annotated

import gymnasium as gym
import numpy as np
import torch
import typer

import simple.envs  # noqa: F401  (registers all `simple/...` env ids, incl. WidowXAITabletopGraspMP-v0)
from simple.core.action import ActionCmd
from simple.robots.protocols import HasKinematics

# Cartesian (position[m], quaternion[wxyz]) + gripper waypoints, empirically pre-verified to have
# a CuRobo IK solution for this robot's cspace with self-collision checking disabled (see module
# docstring) -- picked by probing candidates around the robot's reachable volume, not guessed.
# Position error at each of these was 0.0 m (down to float32 precision) when solved directly by
# `IKSolver.solve_single`, so any error this script measures at runtime reflects the *physics
# settling* step (actuator PD tracking through `env.step()`), not IK inaccuracy.
WAYPOINTS: list[dict] = [
    dict(name="home+up", p=[0.25, 0.00, 0.30], q=[1.0, 0.0, 0.0, 0.0], gripper="open_eef"),
    dict(name="right", p=[0.20, -0.15, 0.20], q=[1.0, 0.0, 0.0, 0.0], gripper="close_eef"),
    dict(name="left", p=[0.20, 0.15, 0.20], q=[1.0, 0.0, 0.0, 0.0], gripper="open_eef"),
    dict(name="down-pitch", p=[0.22, 0.00, 0.15], q=[0.707, 0.0, 0.707, 0.0], gripper="close_eef"),
    dict(name="close-high", p=[0.15, 0.00, 0.35], q=[1.0, 0.0, 0.0, 0.0], gripper="open_eef"),
    dict(name="far", p=[0.40, 0.00, 0.20], q=[1.0, 0.0, 0.0, 0.0], gripper="close_eef"),
    dict(name="gripper-down", p=[0.25, 0.00, 0.20], q=[0.0, 1.0, 0.0, 0.0], gripper="open_eef"),
]


def _solve_ik(
    robot,
    target_p: np.ndarray,
    target_q: np.ndarray,
    current_qpos: np.ndarray,
    self_collision: bool,
) -> np.ndarray:
    """Solve IK for `robot` (a `CuRoboMixin`), toggling self-collision checking.

    `robot.ik()` always builds its solver with self-collision checking ON (see
    `CuRoboMixin._create_empty_world_ik_solver`), which is the setting proven broken for
    widowx_ai in the module docstring above. When `self_collision` is True we go through the
    real `robot.ik()` (to demonstrate the failure with the actual shared code path); when False
    we build an equivalent `IKSolver` by hand with `self_collision_check=False` -- there is no
    public toggle for this on `CuRoboMixin` today.
    """
    if self_collision:
        return robot.ik(target_p, target_q, current_joint=current_qpos)

    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.wrap.reacher.ik_solver import IKSolver

    tensor_args = TensorDeviceType()
    ik_solver = IKSolver(
        IKSolver.load_from_robot_config(
            robot.robot_cfg,
            regularization=True,
            tensor_args=tensor_args,
            self_collision_check=False,
            self_collision_opt=False,
        )
    )
    goal = Pose(
        tensor_args.to_device(torch.as_tensor(target_p, dtype=torch.float32)),
        tensor_args.to_device(torch.as_tensor(target_q, dtype=torch.float32)),
    )
    retract_cfg = torch.as_tensor(current_qpos, dtype=torch.float32).cuda()
    seed_cfg = retract_cfg.unsqueeze(0).repeat(1, 1).cuda().unsqueeze(0)
    result = ik_solver.solve_single(goal, retract_cfg, seed_cfg)
    if not bool(result.success[0, 0]):
        raise RuntimeError("IK failed!")
    return result.solution[0, 0].cpu().numpy()


def _hold_until_enter(env, action: ActionCmd, prompt: str) -> None:
    """Keep stepping `env` with `action` (holding the current commanded pose against gravity)
    until the user presses Enter on stdin. Runs the blocking `input()` call on a background
    thread so the main thread can keep calling `env.step()` -- otherwise the MuJoCo viewer would
    freeze (no physics integration, no `viewer.sync()`) for the whole wait instead of visibly
    holding position.
    """
    advance = threading.Event()

    def _wait_for_enter() -> None:
        try:
            input(prompt)
        except EOFError:
            pass  # no stdin (e.g. piped input) -- fall through and advance immediately
        advance.set()

    thread = threading.Thread(target=_wait_for_enter, daemon=True)
    thread.start()
    while not advance.is_set():
        env.step(action)


def main(
    env_id: Annotated[str, typer.Option(help="Registered gym env id to test against.")] = (
        "simple/WidowXAITabletopGraspMP-v0"
    ),
    scene_uid: Annotated[str, typer.Option(help="Fixed scene, for reproducibility.")] = "hssd:scene1",
    target_object: Annotated[str, typer.Option(help="Fixed graspnet target object id.")] = "graspnet1b:0",
    headless: Annotated[
        bool, typer.Option(help="False opens the MuJoCo viewer so you can watch it move.")
    ] = True,
    settle_steps: Annotated[
        int, typer.Option(help="env.step() calls per waypoint, letting MuJoCo's actuator PD converge.")
    ] = 90,
    pos_tol: Annotated[float, typer.Option(help="Max FK-measured position error (m) counted as a pass.")] = 0.01,
    self_collision: Annotated[
        bool,
        typer.Option(
            "--self-collision/--no-self-collision",
            help=(
                "Use CuRoboMixin's default self-collision-checked IK. Default is OFF because "
                "it is currently broken for widowx_ai (see module docstring) -- pass "
                "--self-collision to reproduce the failure."
            ),
        ),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive/--no-interactive",
            help=(
                "Wait for Enter after each waypoint instead of moving straight to the next one. "
                "Physics keeps stepping (holding the current pose) while waiting. Default off so "
                "the script stays scriptable/CI-friendly by default."
            ),
        ),
    ] = False,
):
    """Drive WidowX AI through fixed Cartesian waypoints via CuRobo IK and measure convergence."""
    env = gym.make(
        env_id,
        sim_mode="mujoco",  # MuJoCo only: no USD exists for WidowX AI (see robots/widowx_ai.py)
        headless=headless,
        scene_uid=scene_uid,
        target_object=target_object,
    )
    env.reset()

    robot = env.unwrapped.task.robot
    assert isinstance(robot, HasKinematics), f"{robot} does not implement HasKinematics"

    n_pass = 0
    for i, wp in enumerate(WAYPOINTS):
        target_p = np.asarray(wp["p"], dtype=np.float32)
        target_q = np.asarray(wp["q"], dtype=np.float32)
        current_qpos = np.asarray(list(robot.get_robot_qpos().values()), dtype=np.float32)

        try:
            ik_qpos = _solve_ik(robot, target_p, target_q, current_qpos, self_collision)
        except RuntimeError as e:
            print(f"[{i}] {wp['name']:14s} IK FAILED: {e}")
            if interactive:
                try:
                    input("  (no pose to hold) Press Enter to continue to the next waypoint... ")
                except EOFError:
                    pass
            continue

        target_qpos = dict(zip(robot.kin_model.joint_names, ik_qpos.tolist()))
        action = ActionCmd("move_qpos_with_eef", target_qpos=target_qpos, eef_state=wp["gripper"])
        for _ in range(settle_steps):
            env.step(action)

        achieved_qpos = np.asarray(list(robot.get_robot_qpos().values()), dtype=np.float32)
        achieved_p, achieved_q = robot.fk(achieved_qpos)
        pos_err = float(np.linalg.norm(achieved_p - target_p))
        ok = pos_err < pos_tol
        n_pass += int(ok)
        status = "OK" if ok else "FAIL (did not settle within tol)"
        print(
            f"[{i}] {wp['name']:14s} target={np.round(target_p, 3)} "
            f"achieved={np.round(achieved_p, 3)} err={pos_err:.4f}m eef={wp['gripper']} -> {status}"
        )

        if interactive and i < len(WAYPOINTS) - 1:
            _hold_until_enter(env, action, "  Holding pose -- press Enter to advance... ")

    env.close()

    print(f"\n{n_pass}/{len(WAYPOINTS)} waypoints converged within {pos_tol} m.")
    sys.exit(0 if n_pass == len(WAYPOINTS) else 1)


if __name__ == "__main__":
    typer.run(main)
