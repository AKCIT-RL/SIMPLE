"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Visualize the Miss platform (ViperX arm + stock gripper + pan/tilt gimbal) moving through its
named joint-space presets in MuJoCo, without needing a task/env (none exists for Miss yet --
Phase 6 of docs/source/miss/miss_integration.md).

WHY A DIRECT MJCF SCRIPT INSTEAD OF gym.make(...)
---------------------------------------------------
Every other `scripts/test_*_curobo_ik.py` script in this repo drives its robot through a real
`gym.make("simple/...")` env, because it needs curobo IK/motion-planning and the task's own
scene/object machinery. This script only needs to move Miss's own joints between named presets
(no Cartesian targets, no grasp planning), so it loads `miss.xml` directly with MuJoCo and drives
`ARM_PRESETS`/`GRIPPER_PRESETS`/`GIMBAL_PRESETS` (`src/simple/robots/miss.py`) via position
actuators -- the same presets a future Miss task would use for its own spawn/reset configuration.

WHAT THIS SCRIPT VALIDATES (AND DOES NOT)
--------------------------------------------
- That every preset actually settles at its commanded qpos under the real actuator PD gains
  authored in miss.xml (`--pos-tol` reports settling error per joint group).
- That moving through the full preset sequence does not produce sustained contact-force blowups
  (a coarse "is anything catastrophically wrong" signal) -- NOT a substitute for the
  ncon/self-collision checks already performed directly against `miss.xml`/`curobo/miss.yml`
  during development (see the module docstring history in miss.py's ARM_PRESETS comment).
- It does NOT validate `gimbal_cam`'s orientation or `cube_2x2x2_v1_1.stl`'s scale (both flagged
  as open TODOs in miss.xml's own comments) -- there is no scene/table to look at yet, only the
  robot in an empty world, so a rendered camera view here would not be meaningful either way.

USAGE
-----
Watch it move in the MuJoCo viewer (needs a display):
    .venv/bin/python scripts/test_miss_gimbal_arm.py --no-headless

Fast, no GUI (settling-error report only):
    .venv/bin/python scripts/test_miss_gimbal_arm.py --headless

Slower/faster settling, tighter/looser tolerance:
    .venv/bin/python scripts/test_miss_gimbal_arm.py --settle-steps 400 --pos-tol 0.01

Exit code is 0 iff every preset in the sequence settles within `--pos-tol` of its commanded qpos.
"""

from __future__ import annotations

import time
from typing import Annotated

import mujoco
import typer

from simple.robots.miss import ARM_PRESETS, GIMBAL_PRESETS, GRIPPER_PRESETS
from simple.utils import resolve_data_path

# (arm preset name, gripper preset name, gimbal preset name). "sleep" is deliberately excluded --
# ARM_PRESETS' own docstring documents it as NOT collision-free on Miss's specific mount.
SEQUENCE: list[tuple[str, str, str]] = [
    ("home", "released", "stow"),
    ("home", "grasping", "look_down"),
    ("upright", "home", "look_down_more"),
    ("upright", "released", "stow"),
    ("desk_reach", "released", "look_at_desk"),  # the class-level default pose
    ("home", "home", "stow"),
]


def _apply_preset(data: mujoco.MjData, arm: str, gripper: str, gimbal: str) -> None:
    for jname, val in ARM_PRESETS[arm].items():
        data.actuator(jname).ctrl = val
    data.actuator("left_finger").ctrl = GRIPPER_PRESETS[gripper]
    for jname, val in GIMBAL_PRESETS[gimbal].items():
        data.actuator(jname).ctrl = val


def _settling_error(model: mujoco.MjModel, data: mujoco.MjData, arm: str, gripper: str, gimbal: str) -> float:
    targets = dict(ARM_PRESETS[arm])
    targets["left_finger"] = GRIPPER_PRESETS[gripper]
    targets.update(GIMBAL_PRESETS[gimbal])
    err = 0.0
    for jname, target in targets.items():
        err = max(err, abs(float(data.joint(jname).qpos[0]) - target))
    return err


def main(
    headless: Annotated[
        bool, typer.Option(help="False opens the MuJoCo viewer so you can watch it move.")
    ] = True,
    settle_steps: Annotated[
        int, typer.Option(help="mj_step() calls per preset, letting the PD actuators converge.")
    ] = 800,  # large transitions (e.g. upright -> desk_reach, ~pi rad of waist travel) need this much
    pos_tol: Annotated[
        float,
        typer.Option(
            help=(
                "Max per-joint settling error (rad or m) counted as a pass. These are P-only "
                "position actuators (no integral term, matching every other robot's controller "
                "in this codebase) -- a small nonzero steady-state error against joint friction "
                "is expected, not a bug; 0.03 rad is a reasonable default, not a tuned bound."
            )
        ),
    ] = 0.05,
    real_time: Annotated[
        bool, typer.Option(help="Sleep between steps in --no-headless mode so playback isn't sped up.")
    ] = True,
):
    """Cycle Miss through its named arm/gripper/gimbal presets and report settling error."""
    model = mujoco.MjModel.from_xml_path(resolve_data_path("robots/miss/miss.xml", auto_download=True))
    data = mujoco.MjData(model)
    # Start at a known, confirmed-collision-free configuration (see ARM_PRESETS' docstring)
    # rather than MuJoCo's raw all-zero default (invalid for the gripper fingers, see
    # miss.py/Miss.init_joint_states's own comment).
    _apply_preset(data, "home", "home", "stow")
    for jname, val in {**ARM_PRESETS["home"], "left_finger": GRIPPER_PRESETS["home"], **GIMBAL_PRESETS["stow"]}.items():
        data.joint(jname).qpos[0] = val
    mujoco.mj_forward(model, data)

    viewer_ctx = None
    if not headless:
        from mujoco import viewer as mj_viewer

        viewer_ctx = mj_viewer.launch_passive(model, data)

    n_pass = 0
    try:
        for arm, gripper, gimbal in SEQUENCE:
            _apply_preset(data, arm, gripper, gimbal)
            for _ in range(settle_steps):
                mujoco.mj_step(model, data)
                if viewer_ctx is not None:
                    viewer_ctx.sync()
                    if real_time:
                        time.sleep(model.opt.timestep)
            err = _settling_error(model, data, arm, gripper, gimbal)
            ok = err < pos_tol
            n_pass += int(ok)
            status = "OK" if ok else "FAIL (did not settle within tol)"
            print(f"arm={arm:8s} gripper={gripper:9s} gimbal={gimbal:15s} err={err:.4f} -> {status}")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()

    print(f"\n{n_pass}/{len(SEQUENCE)} presets settled within {pos_tol} tolerance.")
    raise typer.Exit(code=0 if n_pass == len(SEQUENCE) else 1)


if __name__ == "__main__":
    typer.run(main)
