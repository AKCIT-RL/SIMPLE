"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Preview teleop scenarios in MuJoCo without teleoperation agents.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import simple.envs as _  # import all envs
import typer
import tyro
from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

if TYPE_CHECKING:
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"


def _load_sonic_config() -> dict:
    config = tyro.cli(
        SimLoopConfig,
        config=(tyro.conf.ConsolidateSubcommandArgs,),
        args=[],
    )
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


def _fmt(arr: Any) -> str:
    return np.array2string(np.asarray(arr), precision=4, separator=", ")


def _capture_body_q(robot: G1Sonic) -> np.ndarray:
    return robot.mjData.qpos[robot.body_joint_index + robot.qpos_offset - 1].copy()


def _print_scenario_info(
    sonic_env: "SonicLocoManipEnv",
    env_id: str,
    task: Any,
    robot: G1Sonic,
) -> None:
    task_uid = getattr(task, "uid", "<unknown>")
    task_label = getattr(task, "label", "<unknown>")
    dr_level = task.metadata.get("dr_level", "<unknown>")
    instruction = getattr(task, "instruction", "<missing>")

    print("\n" + "=" * 96)
    print("[PreviewTeleopEnv] Scenario Snapshot")
    print("-" * 96)
    print(f"env_id      : {env_id}")
    print(f"task_uid    : {task_uid}")
    print(f"task_label  : {task_label}")
    print(f"dr_level    : {dr_level}")

    print("\nObjects (from mujoco.mj_objects)")
    mj_objects = sonic_env.mujoco.mj_objects
    if not mj_objects:
        print("  (none)")
    else:
        for layout_key, mj_obj in mj_objects.items():
            actor = task.layout.actors.get(layout_key)
            actor_name = getattr(getattr(actor, "asset", None), "name", "<unknown>")
            actor_uid = getattr(getattr(actor, "asset", None), "uid", "<unknown>")
            print(f"  - key={layout_key} | asset_name={actor_name} | asset_uid={actor_uid}")
            print(f"    pos={_fmt(mj_obj.xpos)}")
            print(f"    quat(wxyz)={_fmt(mj_obj.xquat)}")

    robot_pose = robot.prepare_obs()["floating_base_pose"]
    print("\nRobot spawn/base pose")
    print(f"  qpos[:7]={_fmt(robot_pose)}")

    print("\nTable primitives (from layout actors)")
    found_table = False
    for actor_name, actor in task.layout.actors.items():
        if not actor_name.startswith("table"):
            continue
        if not hasattr(actor, "size") or not hasattr(actor, "pose"):
            continue
        found_table = True
        print(
            f"  - {actor_name}: size={_fmt(actor.size)}, "
            f"position={_fmt(actor.pose.position)}, quat={_fmt(actor.pose.quaternion)}"
        )
    if not found_table:
        print("  (none)")

    print("\nTask instruction")
    print(f"  {instruction}")
    print("=" * 96)


class _NonBlockingKeyReader:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_termios: list[Any] | None = None

    def __enter__(self) -> "_NonBlockingKeyReader":
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is None or self._old_termios is None:
            return
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)

    def read_key(self) -> str | None:
        if self._fd is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return os.read(self._fd, 1).decode(errors="ignore")


def main(
    env_id: str = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
    target: str | None = None,
    dr_level: int = 0,
    max_episode_steps: int = 30000,
    render_hz: int = 50,
    step_hz: float = 50.0,
    headless: bool = False,
) -> None:
    assert step_hz > 0.0, "step_hz must be > 0"

    sonic_config = _load_sonic_config()

    print(f"Creating environment: {env_id}")
    env = gym.make(
        env_id,
        sim_mode="mujoco",
        render_hz=render_hz,
        physics_dt=sonic_config["SIMULATE_DT"],
        headless=headless,
        max_episode_steps=max_episode_steps,
        sonic_config=sonic_config,
        target=target,
        dr_level=dr_level,
    )
    sonic_env: "SonicLocoManipEnv" = env.unwrapped  # type: ignore
    task = sonic_env.task
    robot = task.robot
    assert isinstance(robot, G1Sonic)

    control_dt = 1.0 / step_hz

    def _reset_and_report() -> tuple[dict[str, Any], dict[str, Any], bool, np.ndarray | None]:
        obs, info = env.reset()
        band_is_active = bool(robot.elastic_band is not None and robot.elastic_band.enable)
        hold_q: np.ndarray | None = None
        _print_scenario_info(sonic_env, env_id, task, robot)
        print(
            "[PreviewTeleopEnv] reset complete | "
            f"elastic_band_active={band_is_active} | "
            f"requested_dr_level={dr_level} | effective_dr_level={task.metadata.get('dr_level')}"
        )
        return obs, info, band_is_active, hold_q

    observation, info, band_active, hold_q = _reset_and_report()
    print("[PreviewTeleopEnv] Controls: press R to reset scenario, Ctrl+C to exit.")

    try:
        with _NonBlockingKeyReader() as key_reader:
            while True:
                step_start = time.monotonic()

                key = key_reader.read_key()
                if key is not None and key.lower() == "r":
                    print("[PreviewTeleopEnv] manual reset requested")
                    observation, info, band_active, hold_q = _reset_and_report()
                    continue

                if band_active:
                    action = ActionCmd("elastic_band")
                else:
                    if hold_q is None:
                        hold_q = _capture_body_q(robot)
                    action = ActionCmd(
                        "decoupled_wbc",
                        target_q=hold_q,
                        left_hand_q=None,
                        right_hand_q=None,
                    )

                observation, reward, terminated, truncated, info = env.step(action)
                sonic_env.update_viewer()

                if band_active and (robot.elastic_band is None or not robot.elastic_band.enable):
                    band_active = False
                    hold_q = _capture_body_q(robot)
                    print("[PreviewTeleopEnv] elastic band released, holding current pose")

                if terminated or truncated:
                    print(
                        "[PreviewTeleopEnv] episode ended "
                        f"(terminated={terminated}, truncated={truncated}), auto-reset"
                    )
                    observation, info, band_active, hold_q = _reset_and_report()
                    continue

                elapsed = time.monotonic() - step_start
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
    except KeyboardInterrupt:
        print("\n[PreviewTeleopEnv] interrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    typer.run(main)
