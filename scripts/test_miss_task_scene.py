"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Interactively pick where Miss spawns relative to the desk AND drive its arm/gripper/gimbal, in a
live MuJoCo viewer, WITHOUT running any motion planning -- just the real
`simple/MissTabletopGraspMP-v0` scene (HSSD room + desk + banana + Miss). Same interaction model
as miss_interactive_control.py/miss_camera_tuning.py; this script merges "where does the robot
spawn" (miss_interactive_control.py has no scene/desk at all) with "what pose is it in" (this
script originally just held the reset pose).

WHY THIS EXISTS
---------------
`datagen` currently fails before ever finishing a single episode (a real, pre-existing bug in
shared framework code -- `GSNet.load_cached_grasps`, unrelated to this script), so it can't be
used to visually confirm placement or reach. This script sidesteps grasp planning entirely, and
was used to confirm, directly and numerically (not just by eye), two real bugs in
`MissTabletopGraspTaskMP`'s desk placement -- both now fixed, see that class's own docstring for
the full story:
1. At `table_distance=0.65` (chosen only from `gimbal_cam`'s field of view, without checking
   physical clearance), the robot chassis roughly half-overlapped the desk.
2. More fundamentally: the desk was never actually elevated in world coordinates at all --
   `MujocoEngine._build_primitive` hardcodes the `"table"` actor's Z to floor level regardless of
   any `table_height`, so the robot and desk always sat on the same plane. Fixed by switching the
   task to the `table2` mechanism (which has no such override) for the real desk; this script's
   own diagnostics were updated to read `task.layout.scene.table2`, not `.table`.

TWO DIFFERENT KINDS OF "CHANGE" HERE
----------------------------------------
- `distance`/`angle`/`height`/`yaw`/`set` (desk placement) rebuild the WHOLE scene: the robot's
  spawn pose is baked in at scene-generation time (`SpatialDR`), and `base_link` has no free joint
  (by design, see docs/source/miss/miss_integration.md), so there's no DOF to move at runtime --
  changing these closes and reopens the env (and the viewer window flickers/reopens -- normal, not
  a bug), and **resets the robot back to its default pose**.
- `arm`/`gimbal`/`gripper` (robot pose) just update the held action's target on the *current*
  scene instance -- no rebuild, no window flicker, same mechanism `miss_interactive_control.py`
  uses.

COMMANDS
--------
    distance <m>   | d <m>      # table_distance (rebuilds the scene, resets robot pose)
    angle <rad>    | a <rad>    # table_angle (rebuilds the scene)
    height <m>     | h <m>      # table_height (rebuilds the scene)
    yaw <rad>      | y <rad>    # robot_yaw (rebuilds the scene)
    set <distance> <angle> <height> <yaw>   # all four at once (rebuilds the scene)
    arm = [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
    gimbal = [pitch, yaw]        # pitch->head_tilt, yaw->head_pan
    gripper open | closed
    view cam | view free        # switch the viewer to gimbal_cam's own view, or back to free/orbit
    state                        # reprint current values + diagnostics, no change
    help
    quit | exit                  # or Ctrl+D / Ctrl+C

USAGE
-----
    .venv/bin/python scripts/test_miss_task_scene.py
"""

from __future__ import annotations

import threading
import time

import gymnasium as gym
import mujoco
import numpy as np
import typer

import simple.envs  # noqa: F401  (registers simple/MissTabletopGraspMP-v0)
from simple.core.action import ActionCmd
from simple.robots.miss import GRIPPER_PRESETS

# Chassis collision box is 0.42 x 0.31 x 0.184, centered at the robot's own origin
# (data/robots/miss/miss.urdf) -- half-length in X, used only for the printed clearance check.
_CHASSIS_HALF_LENGTH_X = 0.21

CAM_NAME = "gimbal_cam"
ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
GIMBAL_JOINTS = ["head_tilt", "head_pan"]  # order matches the [pitch, yaw] input convention

HELP_TEXT = __doc__.split("COMMANDS\n--------\n", 1)[1].split("\n\nUSAGE", 1)[0]


def _parse(line: str) -> tuple[str, list[str]] | None:
    cleaned = line.replace("=", " ").replace("[", " ").replace("]", " ").replace(",", " ")
    parts = cleaned.split()
    if not parts:
        return None
    return parts[0].lower(), parts[1:]


class SceneExplorer:
    def __init__(self, scene_uid: str, table_distance: float, table_angle: float, table_height: float, robot_yaw: float):
        self.scene_uid = scene_uid
        self.params = dict(
            table_distance=table_distance, table_angle=table_angle,
            table_height=table_height, robot_yaw=robot_yaw,
        )
        self.env = None
        self.lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._hold_action = None
        self.rebuild()
        self._thread.start()

    def rebuild(self) -> None:
        """Close the current env (if any) and rebuild the whole scene with self.params."""
        with self.lock:
            if self.env is not None:
                self.env.close()
            self.env = gym.make(
                "simple/MissTabletopGraspMP-v0",
                sim_mode="mujoco",
                headless=False,
                scene_uid=self.scene_uid,
                **self.params,
            )
            self.env.reset()
            task = self.env.unwrapped.task
            robot = task.robot
            target_qpos = {j: robot.joints[j].qpos[0] for j in ARM_JOINTS}
            self._hold_action = ActionCmd("move_qpos_with_eef", target_qpos=target_qpos)
        self.view_free()

    def _loop(self) -> None:
        while self._running:
            with self.lock:
                dt = self.env.unwrapped.task.metadata.get("physics_dt", 0.002)
                self.env.step(self._hold_action)
            time.sleep(dt)

    def set_arm(self, values: list[float]) -> None:
        with self.lock:
            self._hold_action.parameters["target_qpos"] = dict(zip(ARM_JOINTS, values))

    def set_gimbal(self, pitch: float, yaw: float) -> None:
        with self.lock:
            self._hold_action.parameters["gimbal_qpos"] = {"head_tilt": pitch, "head_pan": yaw}

    def set_gripper(self, is_open: bool) -> None:
        with self.lock:
            self._hold_action.parameters["eef_state"] = "open_eef" if is_open else "close_eef"

    def wait_until_settled(self, joint_names: list[str], targets: dict[str, float], tol: float = 0.05, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                data = self.env.unwrapped.mujoco.mjData
                err = max(abs(data.joint(j).qpos[0] - targets[j]) for j in joint_names)
            if err < tol:
                return True
            time.sleep(0.05)
        return False

    def view_cam(self) -> None:
        with self.lock:
            sim = self.env.unwrapped.mujoco
            cam_id = mujoco.mj_name2id(sim.mjModel, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
            sim.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            sim.viewer.cam.fixedcamid = cam_id

    def view_free(self) -> None:
        with self.lock:
            viewer = self.env.unwrapped.mujoco.viewer
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = 3.0
            viewer.cam.lookat[:] = [self.params["table_distance"] / 2, 0.0, 0.5]
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

    def print_state(self) -> None:
        with self.lock:
            task = self.env.unwrapped.task
            data = self.env.unwrapped.mujoco.mjData
            robot_pos = np.asarray(task.layout.actors["robot"].pose.position)
            target_pos = np.asarray(task.target.pose.position)
            # table2, not table -- table2 is the real, correctly-elevated desk this task actually
            # uses; "table" is a vestigial floor-level box the hssd scene-building path always
            # creates, which MujocoEngine._build_primitive hardcodes to floor height regardless of
            # table_height (see MissTabletopGraspTaskMP's own class docstring for the full story).
            table = getattr(task.layout.scene, "table2", None)
            table_pos = np.asarray(table.pose.position) if table is not None else None
            table_size = np.asarray(table.size) if table is not None else None
            arm_qpos = [round(float(data.joint(j).qpos[0]), 4) for j in ARM_JOINTS]
            gimbal_qpos = [round(float(data.joint(j).qpos[0]), 4) for j in GIMBAL_JOINTS]
            gripper_left = float(data.joint("left_finger").qpos[0])

        print(f"  table_distance={self.params['table_distance']}  table_angle={self.params['table_angle']}  "
              f"table_height={self.params['table_height']}  robot_yaw={self.params['robot_yaw']}")
        print(f"  robot origin (world): {np.round(robot_pos, 4).tolist()}")
        print(f"  arm     = {arm_qpos}   # {ARM_JOINTS}")
        print(f"  gripper: left_finger={gripper_left:.4f}")
        print(f"  gimbal  = {gimbal_qpos}   # [pitch, yaw] = [head_tilt, head_pan]")
        print(f"  target (banana) pose: {np.round(target_pos, 4).tolist()}")
        if table_pos is not None:
            near_edge_x = table_pos[0] - table_size[0] / 2
            clearance = near_edge_x - _CHASSIS_HALF_LENGTH_X - robot_pos[0]
            print(f"  desk center (world):  {np.round(table_pos, 4).tolist()}")
            print(f"  desk size (x, y, z):  {np.round(table_size, 4).tolist()}")
            print(f"  chassis-to-desk clearance (x): {clearance:.4f} m "
                  f"({'OK, no overlap' if clearance > 0 else 'OVERLAP -- robot is on/inside the desk footprint'})")
        else:
            print("  WARNING: task.layout.scene has no .table2 -- can't report desk bounds.")

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        with self.lock:
            self.env.close()


def main(
    scene_uid: str = "hssd:scene1",
    table_distance: float = 1.75,
    table_angle: float = 0.0,
    table_height: float = 0.70,
    robot_yaw: float = 0.0,
):
    """Interactively adjust Miss's desk placement and arm/gripper/gimbal pose, live."""
    explorer = SceneExplorer(scene_uid, table_distance, table_angle, table_height, robot_yaw)
    print("Miss task-scene explorer -- live MuJoCo viewer running.")
    print(HELP_TEXT)
    print("\nCurrent state:")
    explorer.print_state()

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
                explorer.print_state()
                continue
            elif cmd == "view":
                if not args or args[0].lower() not in ("cam", "free"):
                    print("  ERROR: view needs 'cam' or 'free'")
                    continue
                explorer.view_cam() if args[0].lower() == "cam" else explorer.view_free()
                continue
            elif cmd == "arm":
                if len(args) != len(ARM_JOINTS):
                    print(f"  ERROR: arm needs {len(ARM_JOINTS)} values {ARM_JOINTS}")
                    continue
                try:
                    values = [float(a) for a in args]
                except ValueError:
                    print("  ERROR: all arm values must be numbers")
                    continue
                explorer.set_arm(values)
                explorer.wait_until_settled(ARM_JOINTS, dict(zip(ARM_JOINTS, values)))
                explorer.print_state()
                continue
            elif cmd == "gimbal":
                if len(args) != 2:
                    print("  ERROR: gimbal needs 2 values [pitch, yaw]")
                    continue
                try:
                    pitch, yaw = float(args[0]), float(args[1])
                except ValueError:
                    print("  ERROR: gimbal values must be numbers")
                    continue
                explorer.set_gimbal(pitch, yaw)
                explorer.wait_until_settled(GIMBAL_JOINTS, {"head_tilt": pitch, "head_pan": yaw})
                explorer.print_state()
                continue
            elif cmd == "gripper":
                if not args or args[0].lower() not in ("open", "closed", "close"):
                    print("  ERROR: gripper needs 'open' or 'closed'")
                    continue
                is_open = args[0].lower() == "open"
                explorer.set_gripper(is_open)
                target = GRIPPER_PRESETS["released"] if is_open else GRIPPER_PRESETS["grasping"]
                explorer.wait_until_settled(["left_finger"], {"left_finger": target})
                explorer.print_state()
                continue
            elif cmd in ("distance", "d"):
                key = "table_distance"
            elif cmd in ("angle", "a"):
                key = "table_angle"
            elif cmd in ("height", "h"):
                key = "table_height"
            elif cmd in ("yaw", "y"):
                key = "robot_yaw"
            elif cmd == "set":
                if len(args) != 4:
                    print("  ERROR: set needs 4 values: distance angle height yaw")
                    continue
                try:
                    values = [float(a) for a in args]
                except ValueError:
                    print("  ERROR: all values must be numbers")
                    continue
                explorer.params.update(zip(["table_distance", "table_angle", "table_height", "robot_yaw"], values))
                explorer.rebuild()
                explorer.print_state()
                continue
            else:
                print(f"  ERROR: unknown command '{cmd}'. Type 'help' for the command list.")
                continue

            if len(args) != 1:
                print(f"  ERROR: {cmd} needs exactly 1 value")
                continue
            try:
                explorer.params[key] = float(args[0])
            except ValueError:
                print("  ERROR: value must be a number")
                continue
            explorer.rebuild()
            explorer.print_state()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nClosing...")
        explorer.close()


if __name__ == "__main__":
    typer.run(main)
