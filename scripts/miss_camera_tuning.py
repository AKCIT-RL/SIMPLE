"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Interactive tuning of gimbal_cam's (the Intel RealSense D455) position/orientation in a live
MuJoCo viewer -- type new values, watch the view update immediately, read back what you set. Same
interaction model as miss_interactive_control.py, applied to the camera instead of the joints,
because gimbal_cam's pose is a genuinely open TODO (see docs/source/miss/miss_integration.md):
realsense2_description, which would define the D455's real optical-frame transform, is not
vendored in this checkout, and two guessed 180-degree-flip orientations were already tried and
reverted (both looked at nearby mount hardware, not the desk/workspace).

ONE WINDOW, TWO VIEWS
----------------------
No extra GUI library is used -- MuJoCo's own passive viewer can render through any camera defined
in the model, not just its default free-look camera (`viewer.cam.type = mjCAMERA_FIXED`,
`viewer.cam.fixedcamid = <gimbal_cam's id>`). This script starts the viewer already looking
through gimbal_cam (the view that actually matters for tuning); `view free` switches to a normal
orbiting third-person view for context (where the camera physically is, which way it points),
`view cam` switches back. (An OpenCV window was tried first and dropped: many `opencv-python`
builds, including nix-packaged ones, ship without any GUI backend -- `cv2.namedWindow` raises
"function is not implemented" -- and fixing that is a system-dependency problem, not a SIMPLE one.
Reusing the already-working MuJoCo viewer sidesteps it entirely.)

HOW VALUES MAP TO miss.xml
-----------------------------
`pos`/`quat` here are set directly on the compiled model's `cam_pos`/`cam_quat` arrays, which are
body-relative (relative to `gimbal_head`, the camera's parent body) -- exactly the same
`pos`/`quat` attributes on `miss.xml`'s `<camera name="gimbal_cam" ...>` element. Whatever this
script prints as the final answer can be pasted directly into that line.

COMMANDS
--------
    pos = [x, y, z]                 # absolute position, relative to gimbal_head
    quat = [w, x, y, z]             # absolute orientation (wxyz)
    rpy = [roll, pitch, yaw]        # absolute orientation, degrees, alternative to quat
    nudge <x|y|z> <delta>           # relative position move, e.g. "nudge x 0.02"
    tilt <x|y|z> <deg>              # relative rotation about the camera's OWN current axis,
                                     # e.g. "tilt x 10" tilts the view (typically down/up
                                     # depending on sign -- watch the viewer)
    view cam | view free            # switch the viewer between gimbal_cam's own view (default)
                                     # and a free third-person view for context
    reset                           # back to miss.xml's current on-disk value
    state                           # reprint current pos/quat/rpy, no change
    help
    quit | exit                     # or Ctrl+D / Ctrl+C

`=` and `[`/`]`/`,` are all optional, same as miss_interactive_control.py.

USAGE
-----
    .venv/bin/python scripts/miss_camera_tuning.py

Needs a display for the MuJoCo viewer (X11/Wayland on Linux).
"""

from __future__ import annotations

import threading
import time

import mujoco
import numpy as np
import transforms3d as t3d
from mujoco import viewer as mj_viewer

from simple.robots.miss import ARM_PRESETS, GIMBAL_PRESETS, GRIPPER_PRESETS
from simple.utils import resolve_data_path

CAM_NAME = "gimbal_cam"

HELP_TEXT = __doc__.split("COMMANDS\n--------\n", 1)[1].split("\n\n`=`", 1)[0]


def _parse(line: str) -> tuple[str, list[str]] | None:
    cleaned = line.replace("=", " ").replace("[", " ").replace("]", " ").replace(",", " ")
    parts = cleaned.split()
    if not parts:
        return None
    return parts[0].lower(), parts[1:]


def _quat_to_rpy_deg(quat: np.ndarray) -> list[float]:
    roll, pitch, yaw = t3d.euler.quat2euler(quat)
    return [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]


class CameraTuner:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(resolve_data_path("robots/miss/miss.xml", auto_download=True))
        self.data = mujoco.MjData(self.model)
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
        assert self.cam_id >= 0, f"camera '{CAM_NAME}' not found in miss.xml"

        self._original_pos = self.model.cam_pos[self.cam_id].copy()
        self._original_quat = self.model.cam_quat[self.cam_id].copy()

        self.lock = threading.Lock()

        # Pose the robot so the camera is actually looking toward the desk-reach workspace, not
        # some arbitrary rest pose -- matches the class-level default (see miss.py).
        targets = {**ARM_PRESETS["desk_reach"], "left_finger": GRIPPER_PRESETS["released"]}
        targets.update(GIMBAL_PRESETS["look_at_desk"])
        for jname, val in targets.items():
            self.data.joint(jname).qpos[0] = val
            self.data.actuator(jname).ctrl = val
        mujoco.mj_forward(self.model, self.data)

        self._running = True
        self.viewer = mj_viewer.launch_passive(self.model, self.data)
        self.view_cam()  # start looking through gimbal_cam, the view that matters for tuning

        self._physics_thread = threading.Thread(target=self._loop, daemon=True)
        self._physics_thread.start()

    def _loop(self) -> None:
        dt = self.model.opt.timestep
        while self._running:
            step_start = time.monotonic()
            with self.lock:
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            sleep_for = dt - (time.monotonic() - step_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def view_cam(self) -> None:
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.viewer.cam.fixedcamid = self.cam_id

    def view_free(self) -> None:
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = -20

    def set_pos(self, pos: list[float]) -> None:
        with self.lock:
            self.model.cam_pos[self.cam_id] = pos

    def set_quat(self, quat: list[float]) -> None:
        with self.lock:
            self.model.cam_quat[self.cam_id] = quat

    def set_rpy_deg(self, rpy_deg: list[float]) -> None:
        r, p, y = np.radians(rpy_deg)
        quat = t3d.euler.euler2quat(r, p, y)
        self.set_quat(quat.tolist())

    def nudge(self, axis: str, delta: float) -> None:
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        with self.lock:
            self.model.cam_pos[self.cam_id, idx] += delta

    def tilt(self, axis: str, deg: float) -> None:
        axis_vec = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
        delta_quat = t3d.quaternions.axangle2quat(axis_vec, np.radians(deg))
        with self.lock:
            current = self.model.cam_quat[self.cam_id].copy()
            self.model.cam_quat[self.cam_id] = t3d.quaternions.qmult(current, delta_quat)

    def reset(self) -> None:
        with self.lock:
            self.model.cam_pos[self.cam_id] = self._original_pos
            self.model.cam_quat[self.cam_id] = self._original_quat

    def print_state(self) -> None:
        with self.lock:
            pos = self.model.cam_pos[self.cam_id].copy()
            quat = self.model.cam_quat[self.cam_id].copy()
        rpy = _quat_to_rpy_deg(quat)
        print(f"  pos  = [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
        print(f"  quat = [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]  # wxyz")
        print(f"  rpy  = [{rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}]  # degrees")
        print(f'  -> paste into miss.xml: <camera name="{CAM_NAME}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}" fovy="58"/>')

    def close(self) -> None:
        self._running = False
        self._physics_thread.join(timeout=1.0)
        self.viewer.close()


def main() -> None:
    tuner = CameraTuner()
    print("Miss gimbal_cam tuning -- live MuJoCo viewer running, currently showing gimbal_cam's own view.")
    print(HELP_TEXT)
    print("\nCurrent state (miss.xml's on-disk value):")
    tuner.print_state()

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
                pass
            elif cmd == "reset":
                tuner.reset()
            elif cmd == "view":
                if not args or args[0].lower() not in ("cam", "free"):
                    print("  ERROR: view needs 'cam' or 'free'")
                    continue
                tuner.view_cam() if args[0].lower() == "cam" else tuner.view_free()
                continue
            elif cmd == "pos":
                if len(args) != 3:
                    print("  ERROR: pos needs 3 values [x, y, z]")
                    continue
                try:
                    tuner.set_pos([float(a) for a in args])
                except ValueError:
                    print("  ERROR: pos values must be numbers")
                    continue
            elif cmd == "quat":
                if len(args) != 4:
                    print("  ERROR: quat needs 4 values [w, x, y, z]")
                    continue
                try:
                    tuner.set_quat([float(a) for a in args])
                except ValueError:
                    print("  ERROR: quat values must be numbers")
                    continue
            elif cmd == "rpy":
                if len(args) != 3:
                    print("  ERROR: rpy needs 3 values [roll, pitch, yaw] in degrees")
                    continue
                try:
                    tuner.set_rpy_deg([float(a) for a in args])
                except ValueError:
                    print("  ERROR: rpy values must be numbers")
                    continue
            elif cmd == "nudge":
                if len(args) != 2 or args[0].lower() not in ("x", "y", "z"):
                    print("  ERROR: nudge needs an axis (x|y|z) and a delta, e.g. 'nudge x 0.02'")
                    continue
                try:
                    tuner.nudge(args[0].lower(), float(args[1]))
                except ValueError:
                    print("  ERROR: delta must be a number")
                    continue
            elif cmd == "tilt":
                if len(args) != 2 or args[0].lower() not in ("x", "y", "z"):
                    print("  ERROR: tilt needs an axis (x|y|z) and degrees, e.g. 'tilt x 10'")
                    continue
                try:
                    tuner.tilt(args[0].lower(), float(args[1]))
                except ValueError:
                    print("  ERROR: degrees must be a number")
                    continue
            else:
                print(f"  ERROR: unknown command '{cmd}'. Type 'help' for the command list.")
                continue

            tuner.print_state()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nClosing...")
        tuner.close()


if __name__ == "__main__":
    main()
