#!/usr/bin/env python3
"""Convert a decoupled-WBC teleop LeRobot dataset into the psi0 / gr00t 36-dim format.

`postprocess_psi0.py` targets the older SIMPLE recording schema, whose parquet files
carry `observation.joint_qpos` plus `observation.amo_policy_{command,turning_flag,
target_yaw}`. Datasets recorded through the decoupled-WBC teleop stack use a different
schema and have none of those columns; they carry instead:

    observation.state            (43,)  whole-body joint qpos
    action                       (43,)  whole-body joint targets
    teleop.navigate_command      (4,)   -> psi0 torso_vx / torso_vy / torso_vyaw / target_yaw
    teleop.base_height_command   (1,)   -> psi0 height
    observation.images.ego_view  video

Running `postprocess_psi0.py` on such a dataset is impossible (missing columns), and
simply swapping in a psi0-style modality.json is actively harmful: the slices then land
on unrelated joints, and `action[31]` -- which the SIMPLE deploy path forwards verbatim
as `base_height_command` -- ends up holding an arm joint angle in [-0.36, +0.42] rad
instead of a pelvis height in [0.42, 0.74] m. The WBC drives the pelvis to the floor and
the robot collapses.

IMPORTANT -- the two 43-dim columns do NOT share a joint ordering:

    observation.state : legs 0:12 | waist 12:15 | L_arm 15:22 | R_arm 22:29 | L_hand 29:36 | R_hand 36:43
    action            : legs 0:12 | waist 12:15 | L_arm 15:22 | L_hand 22:29 | R_arm 29:36 | R_hand 36:43

(verified by cross-correlating action[t] against state[t+lag]: action[29:36] tracks
state[22:29] at r = 0.95-0.99, while action[22:29] is identically zero.) `--verify-layout`
re-checks this on the data being converted and refuses to run if it does not hold.

Usage:
    python scripts/postprocess_psi0_teleop_wbc.py \
        --sim-root /path/to/G1WholebodyOpenOvenTeleop-v0 \
        --out-dir  /path/to/out/G1WholebodyOpenOvenTeleop-v0-psi0
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# --- source layout (43-dim whole-body vectors) --------------------------------------
STATE_BLOCKS = {"legs": 0, "waist": 12, "l_arm": 15, "r_arm": 22, "l_hand": 29, "r_hand": 36}
ACTION_BLOCKS = {"legs": 0, "waist": 12, "l_arm": 15, "l_hand": 22, "r_arm": 29, "r_hand": 36}

# Within a 7-dof dex3 hand block: thumb_0,thumb_1,thumb_2, index_0,index_1, middle_0,middle_1
HAND_THUMB, HAND_INDEX, HAND_MIDDLE = (0, 3), (3, 5), (5, 7)

# WAIST_JOINTS = [waist_yaw, waist_roll, waist_pitch] -> psi0 rpy is [roll, pitch, yaw]
WAIST_YAW, WAIST_ROLL, WAIST_PITCH = 0, 1, 2

SRC_STATE_COL = "observation.state"
SRC_ACTION_COL = "action"
SRC_NAV_COL = "teleop.navigate_command"
SRC_HEIGHT_COL = "teleop.base_height_command"

REQUIRED_COLS = (SRC_STATE_COL, SRC_ACTION_COL, SRC_NAV_COL, SRC_HEIGHT_COL)


def _psi0_hand(block: np.ndarray, base: int) -> np.ndarray:
    """Reorder a raw 7-dof hand block into the psi0 [thumb(3), middle(2), index(2)] order."""
    t0, t1 = HAND_THUMB
    i0, i1 = HAND_INDEX
    m0, m1 = HAND_MIDDLE
    return np.concatenate(
        [
            block[:, base + t0 : base + t1],
            block[:, base + m0 : base + m1],
            block[:, base + i0 : base + i1],
        ],
        axis=1,
    )


def build_psi0_vectors(
    state: np.ndarray,
    action: np.ndarray,
    nav_cmd: np.ndarray,
    height_cmd: np.ndarray,
    initial_rpyh: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the psi0 32-dim state and 36-dim action vectors.

    Returns (states, actions, prev_torso_rpy, prev_height).
    """
    # --- action (36) ---------------------------------------------------------------
    a_left_hand = _psi0_hand(action, ACTION_BLOCKS["l_hand"])            # 7
    a_right_hand = action[:, ACTION_BLOCKS["r_hand"] : ACTION_BLOCKS["r_hand"] + 7]
    a_left_arm = action[:, ACTION_BLOCKS["l_arm"] : ACTION_BLOCKS["l_arm"] + 7]
    a_right_arm = action[:, ACTION_BLOCKS["r_arm"] : ACTION_BLOCKS["r_arm"] + 7]

    waist = ACTION_BLOCKS["waist"]
    a_rpy = np.stack(
        [
            action[:, waist + WAIST_ROLL],
            action[:, waist + WAIST_PITCH],
            action[:, waist + WAIST_YAW],
        ],
        axis=1,
    )                                                                    # 3
    a_height = height_cmd.reshape(-1, 1)                                 # 1
    a_nav = nav_cmd[:, 0:4]                                              # torso_vx/vy/vyaw/target_yaw

    actions = np.concatenate(
        [a_left_hand, a_right_hand, a_left_arm, a_right_arm, a_rpy, a_height, a_nav],
        axis=1,
    ).astype(np.float32)
    assert actions.shape[1] == 36, actions.shape

    # --- previous torso command (what state.rpy / state.height are trained on) ------
    # At deployment the agent feeds back the torso rpy + base height it last commanded
    # (simple/baselines/*_decoupled_wbc.py), so the training state must be the previous
    # commanded value, not a proprioceptive read.
    prev = np.concatenate([a_rpy, a_height], axis=1)
    if initial_rpyh is None:
        first = prev[:1]  # hold: assume the command was already active before t=0
    else:
        first = np.asarray(initial_rpyh, dtype=np.float32).reshape(1, 4)
    prev = np.concatenate([first, prev[:-1]], axis=0).astype(np.float32)
    prev_torso_rpy, prev_height = prev[:, 0:3], prev[:, 3:4]

    # --- state (32) ----------------------------------------------------------------
    s_left_hand = _psi0_hand(state, STATE_BLOCKS["l_hand"])
    s_right_hand = state[:, STATE_BLOCKS["r_hand"] : STATE_BLOCKS["r_hand"] + 7]
    s_left_arm = state[:, STATE_BLOCKS["l_arm"] : STATE_BLOCKS["l_arm"] + 7]
    s_right_arm = state[:, STATE_BLOCKS["r_arm"] : STATE_BLOCKS["r_arm"] + 7]

    states = np.concatenate(
        [s_left_hand, s_right_hand, s_left_arm, s_right_arm, prev_torso_rpy, prev_height],
        axis=1,
    ).astype(np.float32)
    assert states.shape[1] == 32, states.shape

    return states, actions, prev_torso_rpy, prev_height


def _track_score(a: np.ndarray, s: np.ndarray, a_base: int, s_base: int, n: int) -> float | None:
    """Median |r| between an action block and a state block, or None if either is static.

    Channels with no variance carry no evidence (an unused hand, a locked joint), so
    they are skipped rather than counted as a mismatch.
    """
    corrs = []
    for k in range(n):
        aj, sk = a[:, a_base + k], s[:, s_base + k]
        if aj.std() < 1e-6 or sk.std() < 1e-6:
            continue
        corrs.append(abs(np.corrcoef(aj, sk)[0, 1]))
    return float(np.median(corrs)) if corrs else None


def verify_layout(
    state: np.ndarray,
    action: np.ndarray,
    lag: int = 5,
    min_corr: float = 0.8,
    min_margin: float = 0.15,
) -> None:
    """Refuse to convert unless the assumed action/state joint orderings actually hold.

    The discriminating question is where right_arm sits inside the action column: at 22
    (directly after left_arm) or at 29 (after a left-hand slot). Decide it by which
    candidate actually tracks the measured right arm, and require a clear margin over
    the loser.

    Do NOT test this by asserting that the left-hand slot is all zeros. That holds only
    for single-arm tasks such as OpenOven; a handover task commands both hands, and the
    slot is legitimately non-zero.
    """
    a, s = action[:-lag], state[lag:]

    # left_arm and waist share the same offset under either hypothesis, so they are a
    # precondition rather than a discriminator.
    for label, a_base, s_base, n in (
        ("left_arm", ACTION_BLOCKS["l_arm"], STATE_BLOCKS["l_arm"], 7),
        ("waist", ACTION_BLOCKS["waist"], STATE_BLOCKS["waist"], 3),
    ):
        score = _track_score(a, s, a_base, s_base, n)
        if score is not None and score < min_corr:
            raise SystemExit(
                f"layout check failed: action[{a_base}:{a_base + n}] does not track "
                f"state[{s_base}:{s_base + n}] for '{label}' (median |r| = {score:.2f} "
                f"< {min_corr}). Joint ordering differs from the assumption baked into "
                f"this script."
            )

    # the discriminator: which action offset carries the right arm
    s_r_arm = STATE_BLOCKS["r_arm"]
    cand = {off: _track_score(a, s, off, s_r_arm, 7) for off in (22, 29)}
    scored = {k: v for k, v in cand.items() if v is not None}
    if not scored:
        raise SystemExit(
            "layout check failed: the right arm is static in this episode, so the action "
            "joint ordering cannot be verified. Re-run pointing at an episode with arm "
            "motion, or pass --no-verify-layout if you have confirmed the layout yourself."
        )

    best = max(scored, key=scored.get)
    runner_up = max((v for k, v in scored.items() if k != best), default=0.0)
    if scored[best] < min_corr or (scored[best] - runner_up) < min_margin:
        raise SystemExit(
            f"layout check failed: cannot tell where the right arm sits in the action "
            f"column. Median |r| against state[{s_r_arm}:{s_r_arm + 7}] per candidate: "
            f"{ {k: round(v, 3) for k, v in scored.items()} }. Need the winner >= "
            f"{min_corr} with a margin >= {min_margin} over the runner-up."
        )
    if best != ACTION_BLOCKS["r_arm"]:
        raise SystemExit(
            f"layout check failed: right arm detected at action[{best}:{best + 7}], but "
            f"this script maps it from action[{ACTION_BLOCKS['r_arm']}:"
            f"{ACTION_BLOCKS['r_arm'] + 7}]. Candidates: "
            f"{ {k: round(v, 3) for k, v in scored.items()} }."
        )
    print(f"  right arm confirmed at action[{best}:{best + 7}] "
          f"(median |r| per candidate: { {k: round(v, 3) for k, v in scored.items()} })")

    # cross-check: the remaining block should be the left hand. Only verifiable when the
    # task actually uses it, which is exactly the case that used to trip the old check.
    l_hand = ACTION_BLOCKS["l_hand"]
    lh = _track_score(a, s, l_hand, STATE_BLOCKS["l_hand"], 7)
    if lh is None:
        print(f"  left hand at action[{l_hand}:{l_hand + 7}] is static (task does not "
              f"use it); nothing to cross-check")
    elif lh < min_corr:
        raise SystemExit(
            f"layout check failed: action[{l_hand}:{l_hand + 7}] does not track the "
            f"measured left hand at state[{STATE_BLOCKS['l_hand']}:"
            f"{STATE_BLOCKS['l_hand'] + 7}] (median |r| = {lh:.2f} < {min_corr})."
        )
    else:
        print(f"  left hand confirmed at action[{l_hand}:{l_hand + 7}] (median |r| = {lh:.2f})")


def modality_dict() -> dict:
    def entry(start, end, original_key, absolute=True):
        return {
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": absolute,
            "dtype": "float32",
            "original_key": original_key,
        }

    return {
        "state": {
            "left_hand": entry(0, 7, "states"),
            "right_hand": entry(7, 14, "states"),
            "left_arm": entry(14, 21, "states"),
            "right_arm": entry(21, 28, "states"),
            "rpy": entry(28, 31, "states"),
            "height": entry(31, 32, "states"),
        },
        "action": {
            "left_hand": entry(0, 7, "action"),
            "right_hand": entry(7, 14, "action"),
            "left_arm": entry(14, 21, "action"),
            "right_arm": entry(21, 28, "action"),
            "rpy": entry(28, 31, "action"),
            "height": entry(31, 32, "action"),
            "torso_vx": entry(32, 33, "action", absolute=False),
            "torso_vy": entry(33, 34, "action", absolute=False),
            "torso_vyaw": entry(34, 35, "action", absolute=False),
            "target_yaw": entry(35, 36, "action"),
        },
        "video": {"rs_view": {"original_key": "observation.images.egocentric"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def stats_block(arr) -> dict:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return {
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": arr.std(axis=0).astype(np.float32).tolist(),
        "min": arr.min(axis=0).astype(np.float32).tolist(),
        "max": arr.max(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(np.float32).tolist(),
    }


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    def default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, separators=(",", ":"), default=default)
            f.write("\n")


def write_video(src: Path, dst: Path, skip: int, downsample: int, fps: int) -> None:
    if skip == 0 and downsample == 1:
        shutil.copyfile(src, dst)
        return
    vf = f"select='gte(n\\,{skip})*not(mod(n-{skip}\\,{downsample}))',setpts=N/({fps}*TB)"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-vf", vf,
           "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)]
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required when --skip/--downsample are set") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {src} -> {dst} (exit {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sim-root", required=True, help="path or glob of source dataset root(s)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip", type=int, default=0, help="drop the first N frames of each episode")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--total-episodes", type=int, default=10**9)
    parser.add_argument("--fps", type=int, default=None, help="defaults to the source fps / downsample")
    parser.add_argument("--video-key", default="observation.images.ego_view")
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument(
        "--initial-rpyh",
        type=float,
        nargs=4,
        default=None,
        metavar=("ROLL", "PITCH", "YAW", "HEIGHT"),
        help="value of the 'previous' torso command at t=0 (default: hold the t=0 command)",
    )
    parser.add_argument("--no-verify-layout", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    for sub in ("data", "videos", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    episode_idx = 0
    last_index = 0
    total_frames = 0
    all_tasks: list[dict] = []
    episodes: list[dict] = []
    episode_stats_rows: list[dict] = []
    acc = {k: [] for k in ("states", "action", "hand", "arm", "leg", "rpy", "height", "timestamp")}
    sim_info = None
    video_shape = None

    for sim_root in sorted(Path(p).resolve() for p in glob.glob(args.sim_root)):
        if episode_idx >= args.total_episodes:
            break
        print(f"Converting: {sim_root}")

        sim_info = json.loads((sim_root / "meta" / "info.json").read_text())
        src_fps = float(sim_info["fps"])
        fps = args.fps if args.fps is not None else int(round(src_fps / args.downsample))
        feature_shape = sim_info.get("features", {}).get(args.video_key, {}).get("shape")
        if feature_shape:
            video_shape = feature_shape

        task_remap = {}
        for t in load_jsonl(sim_root / "meta" / "tasks.jsonl"):
            for existing in all_tasks:
                if existing["task"] == t["task"]:
                    task_remap[t["task_index"]] = existing["task_index"]
                    break
            else:
                all_tasks.append({"task_index": len(all_tasks), "task": t["task"]})
                task_remap[t["task_index"]] = len(all_tasks) - 1
        task_text = {t["task_index"]: t["task"] for t in all_tasks}

        env_configs = {e["episode_index"]: e for e in load_jsonl(sim_root / "meta" / "episodes.jsonl")}

        for data_path in tqdm(sorted((sim_root / "data").glob("chunk-*/episode_*.parquet"))):
            if episode_idx >= args.total_episodes:
                break

            table = pq.read_table(data_path)
            missing = [c for c in REQUIRED_COLS if c not in table.column_names]
            if missing:
                raise SystemExit(
                    f"{data_path}: missing required columns {missing}. This dataset is not in "
                    f"the decoupled-WBC teleop schema; available: {table.column_names}"
                )

            state = np.asarray(table[SRC_STATE_COL].to_pylist(), dtype=np.float32)
            action = np.asarray(table[SRC_ACTION_COL].to_pylist(), dtype=np.float32)
            nav_cmd = np.asarray(table[SRC_NAV_COL].to_pylist(), dtype=np.float32)
            height_cmd = np.asarray(table[SRC_HEIGHT_COL].to_pylist(), dtype=np.float32)

            if episode_idx == 0 and not args.no_verify_layout:
                verify_layout(state, action)
                print("  layout check passed (action/state joint orderings as assumed)")

            states, actions, prev_rpy, prev_h = build_psi0_vectors(
                state, action, nav_cmd, height_cmd, args.initial_rpyh
            )

            # extra proprio columns, kept for parity with the reference psi0 datasets
            hand_joints = state[:, STATE_BLOCKS["l_hand"] : STATE_BLOCKS["r_hand"] + 7]   # 14
            arm_joints = state[:, STATE_BLOCKS["l_arm"] : STATE_BLOCKS["r_arm"] + 7]      # 14
            leg_joints = state[:, 0 : STATE_BLOCKS["waist"] + 3]                          # 15

            m = states.shape[0]
            if m <= args.skip:
                print(f"  skipping {data_path.name}: only {m} frames (<= skip={args.skip})")
                continue
            sl = slice(args.skip, None, args.downsample)
            n = len(range(*sl.indices(m)))

            done = np.zeros((n,), dtype=bool)
            done[-1] = True
            frame_index = np.arange(n, dtype=np.int64)
            timestamp = (frame_index / fps).astype(np.float32)
            src_task = int(np.asarray(table["task_index"].to_pylist()).ravel()[0])
            new_task = task_remap[src_task]
            task_index = np.full((n,), new_task, dtype=np.int64)

            out_table = pa.table({
                "states": states[sl].tolist(),
                "action": actions[sl].tolist(),
                "observation.hand_joints": hand_joints[sl].tolist(),
                "observation.arm_joints": arm_joints[sl].tolist(),
                "observation.leg_joints": leg_joints[sl].tolist(),
                "observation.prev_torso_rpy": prev_rpy[sl].tolist(),
                "observation.prev_height": prev_h[sl].tolist(),
                "timestamp": timestamp,
                "frame_index": frame_index,
                "episode_index": np.full((n,), episode_idx, dtype=np.int64),
                "index": np.arange(last_index, last_index + n, dtype=np.int64),
                "task_index": task_index,
                "next.done": done,
            })

            chunk_id = episode_idx // args.chunks_size
            out_data_dir = out_dir / "data" / f"chunk-{chunk_id:03d}"
            out_data_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(out_table, out_data_dir / f"episode_{episode_idx:06d}.parquet")

            src_ep = int(data_path.stem.split("_")[-1])
            src_video = sim_root / "videos" / data_path.parent.name / args.video_key / f"episode_{src_ep:06d}.mp4"
            if not src_video.exists():
                raise SystemExit(f"missing source video: {src_video}")
            dst_video_dir = out_dir / "videos" / f"chunk-{chunk_id:03d}" / "egocentric"
            dst_video_dir.mkdir(parents=True, exist_ok=True)
            write_video(src_video, dst_video_dir / f"episode_{episode_idx:06d}.mp4",
                        args.skip, args.downsample, fps)

            for key, arr in (("states", states[sl]), ("action", actions[sl]),
                             ("hand", hand_joints[sl]), ("arm", arm_joints[sl]),
                             ("leg", leg_joints[sl]), ("rpy", prev_rpy[sl]),
                             ("height", prev_h[sl]), ("timestamp", timestamp)):
                acc[key].append(arr)

            src_meta = env_configs.get(src_ep, {})
            episodes.append({
                "episode_index": episode_idx,
                "tasks": [new_task],
                "length": n,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + n - 1,
                "robot_type": "g1",
                "instruction": {"task_index": new_task, "task": task_text[new_task]},
                **({"environment_config": src_meta["environment_config"]}
                   if "environment_config" in src_meta else {}),
            })
            episode_stats_rows.append({"episode_index": episode_idx, "stats": {
                "states": stats_block(states[sl]), "action": stats_block(actions[sl])}})

            total_frames += n
            last_index += n
            episode_idx += 1

    if episode_idx == 0:
        raise SystemExit("no episodes converted")

    fps = args.fps if args.fps is not None else int(round(float(sim_info["fps"]) / args.downsample))
    video_shape = video_shape or [360, 640, 3]

    (out_dir / "meta" / "modality.json").write_text(json.dumps(modality_dict(), indent=4))
    write_jsonl(out_dir / "meta" / "tasks.jsonl",
                [{**t, "category": "", "description": t["task"]} for t in all_tasks])
    write_jsonl(out_dir / "meta" / "episodes.jsonl", episodes)
    write_jsonl(out_dir / "meta" / "episodes_stats.jsonl", episode_stats_rows)

    cat = {k: np.concatenate(v) for k, v in acc.items()}
    (out_dir / "meta" / "stats.json").write_text(json.dumps({
        "observation.hand_joints": stats_block(cat["hand"]),
        "observation.arm_joints": stats_block(cat["arm"]),
        "observation.leg_joints": stats_block(cat["leg"]),
        "observation.prev_torso_rpy": stats_block(cat["rpy"]),
        "observation.prev_height": stats_block(cat["height"]),
        "states": stats_block(cat["states"]),
        "action": stats_block(cat["action"]),
        "timestamp": stats_block(cat["timestamp"]),
    }, indent=4))

    (out_dir / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1",
        "robot_type": "g1",
        "total_episodes": episode_idx,
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "total_videos": episode_idx,
        "total_chunks": (episode_idx - 1) // args.chunks_size + 1,
        "chunks_size": args.chunks_size,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.egocentric": {
                "dtype": "video", "shape": video_shape,
                "names": ["height", "width", "channels"],
                "info": {"video.fps": float(fps)},
            },
            "observation.hand_joints": {"dtype": "float32", "shape": [14], "names": None},
            "observation.arm_joints": {"dtype": "float32", "shape": [14], "names": None},
            "observation.leg_joints": {"dtype": "float32", "shape": [15], "names": None},
            "observation.prev_torso_rpy": {"dtype": "float32", "shape": [3], "names": None},
            "observation.prev_height": {"dtype": "float32", "shape": [1], "names": None},
            "states": {"dtype": "float32", "shape": [32], "names": None},
            "action": {"dtype": "float32", "shape": [36], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }, indent=4))

    print(f"\nDone: {episode_idx} episodes, {total_frames} frames -> {out_dir}")
    print("Now validate it:")
    print(f"  python scripts/validate_lerobot_modality.py {out_dir} --expect-psi0")


if __name__ == "__main__":
    main()
