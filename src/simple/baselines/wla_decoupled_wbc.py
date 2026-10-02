"""
UnifoLM-WLA client for the decoupled-WBC (SONIC) eval path.

Talks to scripts/wla/simple_wla_server.py (Psi0 repo) over the SIMPLE HttpActionClient protocol.
Sends the head camera + raw `joint_qpos` (the server builds the 60D WLA state with FK); receives a
chunk of Psi0-layout 36D actions @30 FPS (EE->IK, fig6d->hand joints already done server side) and
executes it through SONIC exactly like psi0_decoupled_wbc.

Env knobs (all optional):
  WLA_EXEC_STEPS     30-FPS chunk steps executed before re-querying (default 30 = full chunk)
  WLA_ORACLE_DIR     dir with <episode_id>.npy = Psi0 actions @30 FPS (M,36); sent as gt_action for the server's
                     --oracle modes (no model). Episode id comes from SIMPLE_EVAL_EPISODE_ID (set by the eval CLI).
"""

import os
import time

import numpy as np

from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.baselines.client import HttpActionClient
from simple.baselines.psi0_decoupled_wbc import from_psi0_upper_joints
from simple.core.action import ActionCmd

CHUNK_FPS = 30.0
CONTROL_FPS = 50.0
CHUNK_LEN = 30


def upsample_chunk(chunk: np.ndarray, n_exec: int) -> np.ndarray:
    """(T,36) @30 FPS -> (ceil(n_exec*50/30),36) @50 FPS, interpolação linear; target_yaw (col 35) com unwrap."""
    chunk = np.asarray(chunk, dtype=np.float64).copy()
    chunk[:, 35] = np.unwrap(chunk[:, 35])
    n_exec = min(n_exec, chunk.shape[0])
    t_src = np.arange(chunk.shape[0]) / CHUNK_FPS
    n_out = int(np.ceil(n_exec * CONTROL_FPS / CHUNK_FPS))
    t_out = np.minimum(np.arange(n_out) / CONTROL_FPS, t_src[-1])
    out = np.stack([np.interp(t_out, t_src, chunk[:, c]) for c in range(chunk.shape[1])], axis=1)
    out[:, 35] = (out[:, 35] + np.pi) % (2 * np.pi) - np.pi
    return out


class WlaDecoupledWbcAgent(SonicDecoupledWbcAgent):
    def __init__(self, robot, host: str, port: int, **kwargs):
        super().__init__(robot, **kwargs)
        self.client = HttpActionClient(host, port)
        self._exec_steps = int(os.environ.get("WLA_EXEC_STEPS", CHUNK_LEN))
        self._oracle_dir = os.environ.get("WLA_ORACLE_DIR")
        self._oracle = None
        self._chunk_idx = 0  # índice (30 FPS) do próximo chunk no episódio
        self._reset_history = True
        indices = self._dwbc_robot_model.get_joint_group_indices("upper_body")
        self.sonic_upper_joint_names = [
            n for n, i in self._dwbc_robot_model.joint_to_dof_index.items() if i in indices
        ]

    def _load_oracle(self):
        ep = os.environ.get("SIMPLE_EVAL_EPISODE_ID", "")
        path = os.path.join(self._oracle_dir, f"{ep}.npy")
        self._oracle = np.load(path)
        print(f"[wla-client] oracle actions {path} {self._oracle.shape}")

    def _query(self, observation, instruction):
        base = self.robot.prepare_obs()["floating_base_pose"]  # xyz + quat(wxyz)
        quat = np.asarray(base[3:7], dtype=np.float64)
        yaw = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
        state = {
            "joint_qpos": np.asarray(observation["joint_qpos"], dtype=np.float32)[None],
            "base_quat": quat.astype(np.float32)[None],
            "yaw": np.array([yaw], dtype=np.float32),
        }
        gt = None
        if self._oracle_dir:
            if self._oracle is None:
                self._load_oracle()
            win = self._oracle[self._chunk_idx:self._chunk_idx + CHUNK_LEN]
            if len(win) == 0:
                raise StopIteration("oracle actions exhausted")
            if len(win) < CHUNK_LEN:  # repete a última ação (mesmo clamp do loader do WLA)
                win = np.concatenate([win, np.repeat(win[-1:], CHUNK_LEN - len(win), axis=0)])
            gt = win.astype(np.float32)
        history = {"reset": True} if self._reset_history else {}
        self._reset_history = False
        t0 = time.perf_counter()
        pred, *_ = self.client.query_action(
            {"rgb_head_stereo_left": observation["head_stereo_left"]},
            instruction or "pick up the blue tote from the shelf and bring it to the table.",
            state, {}, history=history, dataset="simple", gt_action=gt,
        )
        print(f"[wla-client] chunk {self._chunk_idx} queried in {time.perf_counter() - t0:.2f}s")
        return np.asarray(pred, dtype=np.float64)

    def get_action(self, observation, instruction=None, info=None, conditions=None, **kwargs):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        if len(self._action_queue) == 0:
            chunk = self._query(observation, instruction)
            steps = upsample_chunk(chunk, self._exec_steps)
            self._chunk_idx += min(self._exec_steps, chunk.shape[0])
            for a in steps:
                target_qpos = dict(zip(self.robot.joint_names[15:], from_psi0_upper_joints(a[:28])))
                waist = {"waist_yaw_joint": a[30], "waist_roll_joint": a[28], "waist_pitch_joint": a[29]}
                self.queue_action(ActionCmd(
                    "vla_cmd",
                    target_upper_body_pose={**target_qpos, **waist},
                    navigate_cmd=a[32:36],
                    base_height_command=a[31:32],
                ))

        action_cmd = super().get_action(observation, instruction, **kwargs)
        if action_cmd.type != "vla_cmd":
            raise ValueError(f"Unexpected action type {action_cmd.type} from queue.")

        proprio = self.robot.prepare_obs()
        self._wbc_policy.set_observation(self._build_wbc_observation(proprio))
        t_now = time.monotonic()
        target_upper = np.array(
            [action_cmd["target_upper_body_pose"][j] for j in self.sonic_upper_joint_names], dtype=np.float32
        )
        self._wbc_policy.set_goal({
            "target_upper_body_pose": target_upper,
            "navigate_cmd": action_cmd["navigate_cmd"],
            "base_height_command": action_cmd["base_height_command"],
            "target_time": t_now + 1 / self._control_frequency,
            "interpolation_garbage_collection_time": t_now - 2 / self._control_frequency,
            "timestamp": t_now,
        })
        wbc_action = self._wbc_policy.get_action(time=t_now)
        self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="left")
        self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="right")
        self._last_pred_action = ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            left_hand_q=self._cached_left_hand_q,
            right_hand_q=self._cached_right_hand_q,
        )
        return self._last_pred_action

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._chunk_idx = 0
        self._oracle = None
