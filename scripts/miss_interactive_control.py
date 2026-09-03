"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Interactive terminal control of Miss's arm/gripper/gimbal in a live MuJoCo viewer -- type target
positions, watch it move, read back the settled joint state. Meant for exploring/defining the
"core" named positions used elsewhere (`ARM_PRESETS`/`GRIPPER_PRESETS`/`GIMBAL_PRESETS` in
src/simple/robots/miss.py), not for running a task -- no curobo, no gym env, direct joint-space
position commands only.

HOW IT WORKS
------------
Physics stepping and viewer sync run continuously on a background thread (real-time-paced) so the
simulation stays alive and visibly responsive while you're typing at the prompt. The main thread
just reads commands from stdin, writes new actuator targets (under a lock shared with the physics
thread), waits for the arm/gripper/gimbal to settle (or times out), and prints the achieved qpos
in the same list format you typed -- so a position you like can be copied straight into
ARM_PRESETS/GIMBAL_PRESETS.

COMMANDS
--------
    arm = [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
    gimbal = [pitch, yaw]          # pitch -> head_tilt, yaw -> head_pan (see note below)
    gripper open | closed
    state                          # just reprint current state, no change
    help                           # reprint this command list
    quit | exit                    # or Ctrl+D / Ctrl+C

`=`, `[`, `]`, and `,` are all optional -- `arm 0 0.2 -0.5 0 0.3 0` works the same as
`arm = [0, 0.2, -0.5, 0, 0.3, 0]`. Any command can be run again to issue a new target at any time,
including before the previous one has fully settled.

WHY GIMBAL TAKES [pitch, yaw], NOT [head_pan, head_tilt]
-----------------------------------------------------------
Requested in that order; the actual MJCF joint names are head_pan (yaw-like, axis 0 0 -1) and
head_tilt (pitch-like, axis 0 -1 0) -- this script accepts [pitch, yaw] and maps
pitch->head_tilt, yaw->head_pan internally, and the printed state always echoes both the
[pitch, yaw] list and the underlying joint names so the mapping is never ambiguous.

USAGE
-----
    .venv/bin/python scripts/miss_interactive_control.py
"""

from __future__ import annotations

import threading
import time

import mujoco
from mujoco import viewer as mj_viewer

from simple.robots.miss import ARM_PRESETS, GIMBAL_PRESETS, GRIPPER_PRESETS
from simple.utils import resolve_data_path

ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
GIMBAL_JOINTS = ["head_tilt", "head_pan"]  # order matches the [pitch, yaw] input convention
GRIPPER_OPEN = GRIPPER_PRESETS["released"]
GRIPPER_CLOSED = GRIPPER_PRESETS["grasping"]

HELP_TEXT = __doc__.split("COMMANDS\n--------\n", 1)[1].split("\n\nWHY GIMBAL", 1)[0]


def _parse(line: str) -> tuple[str, list[str]] | None:
    cleaned = line.replace("=", " ").replace("[", " ").replace("]", " ").replace(",", " ")
    parts = cleaned.split()
    if not parts:
        return None
    return parts[0].lower(), parts[1:]


class MissInteractive:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(resolve_data_path("robots/miss/miss.xml", auto_download=True))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()

        # Start at the class-level default (Home arm / Home gripper / stow gimbal), the one
        # combination confirmed collision-free in Miss's own docs -- not MuJoCo's raw all-zero
        # default (invalid for the gripper fingers).
        self.targets: dict[str, float] = {**ARM_PRESETS["home"], "left_finger": GRIPPER_PRESETS["home"]}
        self.targets.update(GIMBAL_PRESETS["stow"])
        for jname, val in self.targets.items():
            self.data.joint(jname).qpos[0] = val
            self.data.actuator(jname).ctrl = val
        mujoco.mj_forward(self.model, self.data)

        self._running = True
        self.viewer = mj_viewer.launch_passive(self.model, self.data)
        self._physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self._physics_thread.start()

    def _physics_loop(self) -> None:
        dt = self.model.opt.timestep
        while self._running:
            step_start = time.monotonic()
            with self.lock:
                for jname, val in self.targets.items():
                    self.data.actuator(jname).ctrl = val
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            sleep_for = dt - (time.monotonic() - step_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def set_targets(self, updates: dict[str, float]) -> None:
        with self.lock:
            self.targets.update(updates)

    def wait_until_settled(self, joint_names: list[str], tol: float = 0.02, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                err = max(abs(self.data.joint(j).qpos[0] - self.targets[j]) for j in joint_names)
            if err < tol:
                return True
            time.sleep(0.05)
        return False

    def print_state(self) -> None:
        with self.lock:
            arm_qpos = [round(float(self.data.joint(j).qpos[0]), 4) for j in ARM_JOINTS]
            gripper_left = float(self.data.joint("left_finger").qpos[0])
            gripper_right = float(self.data.joint("right_finger").qpos[0])
            gimbal_qpos = [round(float(self.data.joint(j).qpos[0]), 4) for j in GIMBAL_JOINTS]
        print(f"  arm     = {arm_qpos}   # {ARM_JOINTS}")
        print(f"  gripper: left_finger={gripper_left:.4f} right_finger={gripper_right:.4f}")
        print(f"  gimbal  = {gimbal_qpos}   # [pitch, yaw] = [head_tilt, head_pan]")

    def close(self) -> None:
        self._running = False
        self._physics_thread.join(timeout=1.0)
        self.viewer.close()


def main() -> None:
    sim = MissInteractive()
    print("Miss interactive control -- live MuJoCo viewer running on a background thread.")
    print(HELP_TEXT)
    print("\nCurrent state:")
    sim.print_state()

    try:
        while True:
            try:
                line = input("\n> ")
            except EOFError:
                break

            parsed = _parse(line)
            if parsed is None:
                continue
            cmd, args = parsed

            if cmd in ("quit", "exit"):
                break
            elif cmd == "help":
                print(HELP_TEXT)
                continue
            elif cmd == "state":
                sim.print_state()
                continue
            elif cmd == "arm":
                if len(args) != len(ARM_JOINTS):
                    print(f"  ERROR: arm needs {len(ARM_JOINTS)} values {ARM_JOINTS}, got {len(args)}")
                    continue
                try:
                    values = [float(a) for a in args]
                except ValueError:
                    print("  ERROR: all arm values must be numbers")
                    continue
                updates = dict(zip(ARM_JOINTS, values))
                sim.set_targets(updates)
                settled = sim.wait_until_settled(ARM_JOINTS)
            elif cmd == "gimbal":
                if len(args) != 2:
                    print("  ERROR: gimbal needs 2 values [pitch, yaw]")
                    continue
                try:
                    pitch, yaw = float(args[0]), float(args[1])
                except ValueError:
                    print("  ERROR: gimbal values must be numbers")
                    continue
                updates = {"head_tilt": pitch, "head_pan": yaw}
                sim.set_targets(updates)
                settled = sim.wait_until_settled(GIMBAL_JOINTS)
            elif cmd == "gripper":
                if not args or args[0].lower() not in ("open", "closed", "close"):
                    print("  ERROR: gripper needs 'open' or 'closed'")
                    continue
                is_open = args[0].lower() == "open"
                sim.set_targets({"left_finger": GRIPPER_OPEN if is_open else GRIPPER_CLOSED})
                settled = sim.wait_until_settled(["left_finger"])
            else:
                print(f"  ERROR: unknown command '{cmd}'. Type 'help' for the command list.")
                continue

            if not settled:
                print("  (did not fully settle within timeout -- showing current state anyway)")
            sim.print_state()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nClosing viewer...")
        sim.close()


if __name__ == "__main__":
    main()
