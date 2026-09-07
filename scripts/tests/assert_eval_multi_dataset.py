#!/usr/bin/env python3
"""Assertions for check_eval_multi_dataset.sh.

Kept out of the shell script because these checks compare *sets* of episodes
across runs, which bash does poorly. Reads only the artifacts the eval left
behind, so it can also be pointed at an older run directory:

    EVAL_ROOT=.test-output/eval-multi-dataset-... \
    ENV_ID_TAIL=G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop-v0 \
    NUM_EPISODES=6 python scripts/tests/assert_eval_multi_dataset.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EVAL_ROOT = Path(os.environ["EVAL_ROOT"])
ENV_ID_TAIL = os.environ["ENV_ID_TAIL"]
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "6"))

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def episode_lines(eval_dir: Path) -> list[tuple[str, str]]:
    """[(task_id, instruction)] from eval_stats.txt, skipping header/summary."""
    stats = eval_dir / "eval_stats.txt"
    if not stats.exists():
        return []
    out = []
    for line in stats.read_text().splitlines():
        if "|" not in line or line.startswith(("=", "run:", "success rate")):
            continue
        head, instruction = line.split("|", 1)
        task_id = head.split(":")[0].strip()
        out.append((task_id, instruction.strip()))
    return out


single = episode_lines(EVAL_ROOT / "single")
parallel = episode_lines(EVAL_ROOT / "parallel")

print("single-worker run:")
check(len(single) == NUM_EPISODES, f"{NUM_EPISODES} episodes evaluated (got {len(single)})")

sources = {task_id.split("__episode_")[0] for task_id, _ in single}
config = (EVAL_ROOT / "eval_sources.yaml").read_text()
declared = {line.split(":", 1)[1].strip() for line in config.splitlines() if "name:" in line}
check(
    bool(sources) and sources == declared,
    f"every declared source appears (declared={sorted(declared)}, seen={sorted(sources)})",
)
check(
    all("__episode_" in task_id for task_id, _ in single),
    "every task_id is prefixed with its source",
)
check(
    len({task_id for task_id, _ in single}) == len(single),
    "no task_id repeats",
)

# The bug this guards: a psi0 dataset stores a task *index* in episodes.jsonl,
# so a naive read hands the policy the integer 0 as its instruction.
requests = []
request_log = EVAL_ROOT / "policy-requests.jsonl"
if request_log.exists():
    requests = [json.loads(line) for line in request_log.read_text().splitlines() if line]
instructions = {r.get("instruction") for r in requests}
check(bool(requests), f"policy received requests ({len(requests)})")
check(
    all(isinstance(i, str) and i.strip() for i in instructions),
    f"every instruction reaching the policy is a non-empty string ({sorted(instructions)[:2]})",
)
logged = {instruction for _, instruction in single}
check(
    instructions <= logged or not logged,
    "instructions sent to the policy match the ones logged in eval_stats.txt",
)

# A `prompt: task` source must NOT be conditioned on its recorded wording. The
# language randomizer's state rides along in dr_state_dict, so replaying an
# episode restores the capturing task's prompt verbatim unless that key is
# dropped -- silently defeating the mode.
import yaml  # noqa: E402

sources = yaml.safe_load(Path(EVAL_ROOT / "eval_sources.yaml").read_text())["datasets"]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from simple.datasets.lerobot import _load_episode_prompts  # noqa: E402

for source in sources:
    lines = [(t, i) for t, i in single if t.startswith(source["name"] + "__episode_")]
    if source.get("prompt") == "task":
        recorded = _load_episode_prompts(source["path"])
        for task_id, instruction in lines:
            idx = int(task_id.rsplit("_", 1)[1])
            check(
                instruction != recorded.get(idx),
                f"{task_id} ignored its recorded prompt (got {instruction!r})",
            )

    # A pinned condition has to show up in the wording the policy is given --
    # that wording is what the task also scores against.
    for key, value in (source.get("conditions") or {}).items():
        allowed = value if isinstance(value, list) else [value]
        if allowed == ["recorded"]:
            continue
        for task_id, instruction in lines:
            hit = [v for v in allowed if str(v) in instruction]
            check(
                bool(hit),
                f"{task_id} prompt reflects a commanded {key} from {allowed} "
                f"(got {instruction!r})",
            )

# VideoRecorder writes one dir per task_id, holding the camera streams. With a
# bare "episode_<n>" id two sources sharing an index would land in the same dir
# and overwrite each other.
video_root = EVAL_ROOT / "single" / "replay_policy" / ENV_ID_TAIL / "train"
video_dirs = {d.name for d in video_root.glob("*") if d.is_dir()} if video_root.exists() else set()
check(
    video_dirs == {task_id for task_id, _ in single},
    f"one video dir per episode, none overwritten (got {sorted(video_dirs)})",
)
check(
    all(any(d.glob("*.mp4")) for d in video_root.glob("*") if d.is_dir()),
    "each video dir holds at least one .mp4",
)

print("two-worker run:")
check(len(parallel) == NUM_EPISODES, f"{NUM_EPISODES} episodes evaluated (got {len(parallel)})")
check(
    {task_id for task_id, _ in parallel} == {task_id for task_id, _ in single},
    "same seed selects the same episodes regardless of worker count",
)
check(
    len({task_id for task_id, _ in parallel}) == len(parallel),
    "workers partitioned the plan instead of duplicating episodes",
)

print("legacy --data-dir run:")
legacy = episode_lines(EVAL_ROOT / "legacy")
legacy_stats = (EVAL_ROOT / "legacy" / "eval_stats.txt")
check(legacy_stats.exists(), "eval_stats.txt written")
check(
    "success rate:" in (legacy_stats.read_text() if legacy_stats.exists() else ""),
    "success rate reported",
)
check(
    all(task_id.startswith("episode_") for task_id, _ in legacy),
    f"task ids keep the old unprefixed form (got {[t for t, _ in legacy]})",
)

print()
if failures:
    print(f"{len(failures)} assertion(s) failed:")
    for message in failures:
        print(f"  - {message}")
    sys.exit(1)
print("all assertions passed")
