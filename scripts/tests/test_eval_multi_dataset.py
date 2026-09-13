"""Unit tests for the --eval-config multi-dataset eval path.

No simulator and no policy server: everything here reads meta/ JSON only, so
the fixtures build fake lerobot roots holding just meta/info.json,
meta/episodes.jsonl and meta/tasks.jsonl. The loader and the plan builder never
touch parquet or video, which is exactly why they can be tested this way.

    uv run pytest scripts/tests/test_eval_multi_dataset.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import typer

from simple.cli.eval_decoupled_wbc import (
    _build_episode_plan,
    _eligible_episodes,
    _load_eval_datasets,
    _resolve_conditions,
)
from simple.datasets.lerobot import _load_episode_prompts, _load_episode_tasks

REPO_ROOT = Path(__file__).resolve().parents[2]

# Real datasets, when present on this machine. Skipped elsewhere so the suite
# stays runnable anywhere.
PSI0_DIR = Path(
    os.environ.get(
        "SIMPLE_TEST_PSI0_DIR",
        Path.home() / "workspace/unitree_ws/SIMPLE/data/psi0_G1IndustrialSortingTeleop",
    )
)
RAW_DIR = REPO_ROOT / (
    "data/teleop_decoupled_wbc/simple/"
    "G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop-v0/level-0/"
    "sessions/20260816_210834__gustavo"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_root(
    tmp_path: Path,
    name: str,
    n_episodes: int,
    *,
    fps: int = 50,
    prompt: str = "do the thing",
    task_indices: bool = False,
    missing_env_config: tuple[int, ...] = (),
    with_tasks_jsonl: bool = True,
) -> Path:
    """Build a fake lerobot root with `n_episodes` metadata rows.

    task_indices=True mimics the psi0 layout, where episodes.jsonl stores a task
    *index* instead of the prompt string.
    """
    root = tmp_path / name
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": n_episodes, "fps": fps, "chunks_size": 1000})
    )
    rows = []
    for i in range(n_episodes):
        row: dict = {
            "episode_index": i,
            "tasks": [0 if task_indices else prompt],
        }
        if i not in missing_env_config:
            row["environment_config"] = json.dumps({"dr_state_dict": {}, "ep": i})
        rows.append(row)
    _write_jsonl(root / "meta" / "episodes.jsonl", rows)
    if with_tasks_jsonl:
        _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": prompt}])
    return root


# --------------------------------------------------------------------------
# _load_eval_datasets
# --------------------------------------------------------------------------


def _write_config(tmp_path: Path, body: str) -> str:
    path = tmp_path / "eval.yaml"
    path.write_text(body)
    return str(path)


def test_entry_may_be_a_bare_path_string(tmp_path):
    root = make_root(tmp_path, "alpha", 3)
    cfg = _write_config(tmp_path, f"datasets:\n  - {root}\n")
    assert _load_eval_datasets(cfg) == [("alpha", str(root), "recorded", {})]


def test_entry_mapping_sets_name_and_prompt_mode(tmp_path):
    root = make_root(tmp_path, "alpha", 3)
    cfg = _write_config(
        tmp_path, f"datasets:\n  - name: old\n    path: {root}\n    prompt: task\n"
    )
    assert _load_eval_datasets(cfg) == [("old", str(root), "task", {})]


def test_default_name_skips_the_level_dir(tmp_path):
    root = make_root(tmp_path, "MyTaskEnv-v0/level-0", 2)
    cfg = _write_config(tmp_path, f"datasets:\n  - {root}\n")
    # "level-0" would be the same for every entry, so the dir above it wins.
    assert _load_eval_datasets(cfg)[0][0] == "MyTaskEnv-v0"


def test_colliding_names_are_disambiguated(tmp_path):
    a = make_root(tmp_path, "a/level-0", 2)
    b = make_root(tmp_path, "b/level-0", 2)
    cfg = _write_config(
        tmp_path, f"datasets:\n  - name: same\n    path: {a}\n  - name: same\n    path: {b}\n"
    )
    assert [d[0] for d in _load_eval_datasets(cfg)] == ["same", "same_2"]


def test_path_without_meta_info_is_rejected(tmp_path):
    missing = tmp_path / "not-a-dataset"
    missing.mkdir()
    cfg = _write_config(tmp_path, f"datasets:\n  - {missing}\n")
    with pytest.raises(typer.BadParameter, match="not a lerobot root"):
        _load_eval_datasets(cfg)


def test_unknown_prompt_mode_is_rejected(tmp_path):
    root = make_root(tmp_path, "alpha", 2)
    cfg = _write_config(tmp_path, f"datasets:\n  - path: {root}\n    prompt: invented\n")
    with pytest.raises(typer.BadParameter, match="expected 'recorded' or 'task'"):
        _load_eval_datasets(cfg)


def test_empty_config_is_rejected(tmp_path):
    cfg = _write_config(tmp_path, "datasets: []\n")
    with pytest.raises(typer.BadParameter, match="no 'datasets'"):
        _load_eval_datasets(cfg)


# --------------------------------------------------------------------------
# _build_episode_plan
# --------------------------------------------------------------------------


def _datasets(*roots: Path) -> list[tuple[str, str, str, dict]]:
    return [(p.name, str(p), "recorded", {}) for p in roots]


def _counts(plan) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, *_ in plan:
        out[name] = out.get(name, 0) + 1
    return out


def test_quota_is_split_evenly_regardless_of_dataset_size(tmp_path):
    big = make_root(tmp_path, "big", 120)
    small = make_root(tmp_path, "small", 35)
    plan = _build_episode_plan(_datasets(big, small), num_episodes=20, seed=0)
    assert _counts(plan) == {"big": 10, "small": 10}


def test_odd_quota_remainder_goes_to_the_first_datasets(tmp_path):
    a = make_root(tmp_path, "a", 50)
    b = make_root(tmp_path, "b", 50)
    plan = _build_episode_plan(_datasets(a, b), num_episodes=7, seed=0)
    assert _counts(plan) == {"a": 4, "b": 3}


def test_dataset_smaller_than_its_quota_hands_the_rest_back(tmp_path):
    tiny = make_root(tmp_path, "tiny", 1)
    big = make_root(tmp_path, "big", 100)
    plan = _build_episode_plan(_datasets(tiny, big), num_episodes=10, seed=0)
    assert _counts(plan) == {"tiny": 1, "big": 9}


def test_requesting_more_than_available_uses_everything(tmp_path):
    a = make_root(tmp_path, "a", 3)
    b = make_root(tmp_path, "b", 2)
    plan = _build_episode_plan(_datasets(a, b), num_episodes=99, seed=0)
    assert len(plan) == 5


def test_same_seed_reproduces_the_plan(tmp_path):
    a = make_root(tmp_path, "a", 40)
    b = make_root(tmp_path, "b", 40)
    args = (_datasets(a, b), 12)
    assert _build_episode_plan(*args, seed=0) == _build_episode_plan(*args, seed=0)


def test_different_seed_changes_the_plan(tmp_path):
    a = make_root(tmp_path, "a", 40)
    b = make_root(tmp_path, "b", 40)
    assert _build_episode_plan(_datasets(a, b), 12, seed=0) != _build_episode_plan(
        _datasets(a, b), 12, seed=1
    )


def test_equal_sized_datasets_do_not_draw_identical_indices(tmp_path):
    a = make_root(tmp_path, "a", 40)
    b = make_root(tmp_path, "b", 40)
    plan = _build_episode_plan(_datasets(a, b), num_episodes=20, seed=0)
    picked = {name: sorted(i for n, _, i, _, _ in plan if n == name) for name in ("a", "b")}
    assert picked["a"] != picked["b"]


def test_no_episode_is_scheduled_twice(tmp_path):
    a = make_root(tmp_path, "a", 30)
    b = make_root(tmp_path, "b", 30)
    plan = _build_episode_plan(_datasets(a, b), num_episodes=24, seed=0)
    keys = [(name, idx) for name, _, idx, _, _ in plan]
    assert len(set(keys)) == len(keys) == 24


def test_worker_split_partitions_the_plan(tmp_path):
    a = make_root(tmp_path, "a", 30)
    b = make_root(tmp_path, "b", 30)
    plan = _build_episode_plan(_datasets(a, b), num_episodes=13, seed=0)
    shards = [list(plan)[w::3] for w in range(3)]
    assert sum(len(s) for s in shards) == len(plan)
    assert {item for shard in shards for item in shard} == set(plan)


def test_prompt_mode_rides_along_in_the_plan(tmp_path):
    a = make_root(tmp_path, "a", 5)
    b = make_root(tmp_path, "b", 5)
    datasets = [("a", str(a), "task", {}), ("b", str(b), "recorded", {})]
    plan = _build_episode_plan(datasets, num_episodes=4, seed=0)
    assert {name: mode for name, _, _, mode, _ in plan} == {"a": "task", "b": "recorded"}


def test_mixed_fps_is_rejected(tmp_path):
    a = make_root(tmp_path, "a", 10, fps=50)
    b = make_root(tmp_path, "b", 10, fps=30)
    with pytest.raises(typer.BadParameter, match="disagree on fps"):
        _build_episode_plan(_datasets(a, b), num_episodes=4, seed=0)


def test_zero_episodes_is_rejected(tmp_path):
    a = make_root(tmp_path, "a", 10)
    with pytest.raises(typer.BadParameter, match="must be > 0"):
        _build_episode_plan(_datasets(a), num_episodes=0, seed=0)


# --------------------------------------------------------------------------
# eligibility (environment_config)
# --------------------------------------------------------------------------


def test_episodes_without_environment_config_are_excluded(tmp_path):
    root = make_root(tmp_path, "a", 5, missing_env_config=(1, 3))
    eligible, dropped = _eligible_episodes(str(root))
    assert eligible == [0, 2, 4]
    assert dropped == 2


def test_plan_never_schedules_an_ineligible_episode(tmp_path):
    root = make_root(tmp_path, "a", 5, missing_env_config=(1, 3))
    plan = _build_episode_plan(_datasets(root), num_episodes=5, seed=0)
    assert sorted(idx for _, _, idx, _, _ in plan) == [0, 2, 4]


def test_dataset_with_no_usable_episode_is_rejected(tmp_path):
    root = make_root(tmp_path, "a", 2, missing_env_config=(0, 1))
    with pytest.raises(typer.BadParameter, match="no episodes with an environment_config"):
        _build_episode_plan(_datasets(root), num_episodes=1, seed=0)


# --------------------------------------------------------------------------
# prompt resolution
# --------------------------------------------------------------------------


def test_raw_format_prompt_is_used_verbatim(tmp_path):
    root = make_root(tmp_path, "raw", 3, prompt="bring it to the table.")
    assert _load_episode_prompts(root) == {i: "bring it to the table." for i in range(3)}


def test_psi0_format_task_index_is_resolved_to_a_string(tmp_path):
    root = make_root(tmp_path, "psi0", 3, prompt="put 2 screws in the tote.", task_indices=True)
    prompts = _load_episode_prompts(root)
    assert prompts == {i: "put 2 screws in the tote." for i in range(3)}
    assert all(isinstance(v, str) for v in prompts.values())


def test_task_index_without_tasks_jsonl_yields_no_prompt(tmp_path):
    # Falls through to task.instruction in the eval rather than blowing up.
    root = make_root(tmp_path, "psi0", 2, task_indices=True, with_tasks_jsonl=False)
    assert _load_episode_prompts(root) == {}


def test_missing_metadata_files_return_empty(tmp_path):
    assert _load_episode_prompts(tmp_path / "nope") == {}
    assert _load_episode_tasks(tmp_path / "nope") == {}


# --------------------------------------------------------------------------
# real datasets, when available
# --------------------------------------------------------------------------


@pytest.mark.skipif(not PSI0_DIR.exists(), reason=f"no psi0 dataset at {PSI0_DIR}")
def test_real_psi0_prompts_resolve_to_strings():
    prompts = _load_episode_prompts(PSI0_DIR)
    tasks = set(_load_episode_tasks(PSI0_DIR).values())
    assert prompts, "psi0 dataset yielded no prompts"
    assert all(isinstance(v, str) and v for v in prompts.values())
    assert set(prompts.values()) <= tasks


@pytest.mark.skipif(not PSI0_DIR.exists(), reason=f"no psi0 dataset at {PSI0_DIR}")
def test_real_psi0_episodes_are_all_eligible():
    eligible, dropped = _eligible_episodes(str(PSI0_DIR))
    assert dropped == 0, "postprocess should never emit a row without environment_config"
    assert len(eligible) == json.loads((PSI0_DIR / "meta" / "info.json").read_text())[
        "total_episodes"
    ]


@pytest.mark.skipif(not RAW_DIR.exists(), reason=f"no capture at {RAW_DIR}")
def test_real_capture_prompts_resolve_to_strings():
    prompts = _load_episode_prompts(RAW_DIR)
    assert prompts and all(isinstance(v, str) and v for v in prompts.values())


@pytest.mark.skipif(not PSI0_DIR.exists(), reason=f"no psi0 dataset at {PSI0_DIR}")
def test_load_frames_false_skips_video_decoding():
    """The frames are only consumed by the vlt baseline; building them decodes
    one video frame per step (~4 min for a psi0 episode) for every other policy.
    """
    import time

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from simple.datasets.lerobot import get_episode_lerobot

    ds = LeRobotDataset(repo_id="simple/test", root=str(PSI0_DIR))
    started = time.perf_counter()
    env_conf, frames = get_episode_lerobot(ds, 0, load_frames=False)
    elapsed = time.perf_counter() - started

    assert frames is None
    assert env_conf and "dr_state_dict" in env_conf
    # The episode is thousands of frames long; decoding them takes minutes.
    assert elapsed < 10, f"took {elapsed:.1f}s -- frames are still being decoded"


# --------------------------------------------------------------------------
# per-episode conditions
# --------------------------------------------------------------------------


def _conditions(plan, name: str) -> list[dict]:
    return [dict(c) for n, _, _, _, c in plan if n == name]


def test_scalar_condition_pins_every_episode(tmp_path):
    root = make_root(tmp_path, "old", 10)
    datasets = [("old", str(root), "task", {"pick_hand": "right"})]
    plan = _build_episode_plan(datasets, num_episodes=4, seed=0)
    assert _conditions(plan, "old") == [{"pick_hand": "right"}] * 4


def test_list_condition_is_dealt_round_robin(tmp_path):
    root = make_root(tmp_path, "old", 10)
    datasets = [("old", str(root), "task", {"target_side": ["left", "right"]})]
    plan = _build_episode_plan(datasets, num_episodes=4, seed=0)
    sides = [c["target_side"] for c in _conditions(plan, "old")]
    assert sorted(sides) == ["left", "left", "right", "right"]


def test_conditions_combine_independently(tmp_path):
    root = make_root(tmp_path, "old", 10)
    datasets = [
        ("old", str(root), "task", {"pick_hand": "right", "target_side": ["left", "right"]})
    ]
    plan = _build_episode_plan(datasets, num_episodes=4, seed=0)
    got = _conditions(plan, "old")
    assert all(c["pick_hand"] == "right" for c in got)
    assert sorted(c["target_side"] for c in got) == ["left", "left", "right", "right"]


def test_recorded_condition_writes_no_key(tmp_path):
    # Leaving the key out is what lets the episode's own value survive.
    root = make_root(tmp_path, "mirror", 10)
    datasets = [("mirror", str(root), "recorded", {"target_side": "recorded"})]
    plan = _build_episode_plan(datasets, num_episodes=3, seed=0)
    assert _conditions(plan, "mirror") == [{}] * 3


def test_conditions_are_reproducible(tmp_path):
    root = make_root(tmp_path, "old", 10)
    datasets = [("old", str(root), "task", {"target_side": ["left", "right"]})]
    args = (datasets, 6)
    assert _build_episode_plan(*args, seed=0) == _build_episode_plan(*args, seed=0)


def test_round_robin_position_is_stable_per_episode(tmp_path):
    # The plan is shuffled after the deal, so a condition must stay glued to its
    # episode rather than to a slot in the final ordering.
    root = make_root(tmp_path, "old", 10)
    datasets = [("old", str(root), "task", {"target_side": ["left", "right"]})]
    a = {(i, dict(c)["target_side"]) for _, _, i, _, c in _build_episode_plan(datasets, 6, seed=0)}
    b = {(i, dict(c)["target_side"]) for _, _, i, _, c in _build_episode_plan(datasets, 6, seed=0)}
    assert a == b


def test_resolve_conditions_cycles_by_position():
    spec = {"target_side": ["left", "right"], "pick_hand": "right"}
    assert dict(_resolve_conditions(spec, 0)) == {"target_side": "left", "pick_hand": "right"}
    assert dict(_resolve_conditions(spec, 1)) == {"target_side": "right", "pick_hand": "right"}
    assert dict(_resolve_conditions(spec, 2)) == {"target_side": "left", "pick_hand": "right"}


def test_conditions_are_hashable_so_the_plan_can_be_diffed(tmp_path):
    root = make_root(tmp_path, "old", 10)
    datasets = [("old", str(root), "task", {"target_side": ["left", "right"]})]
    plan = _build_episode_plan(datasets, num_episodes=4, seed=0)
    assert len(set(plan)) == len(plan)  # worker-split assertions rely on this


def test_conditions_parsed_from_yaml(tmp_path):
    root = make_root(tmp_path, "old", 5)
    cfg = _write_config(
        tmp_path,
        f"datasets:\n  - path: {root}\n    prompt: task\n"
        "    conditions:\n      pick_hand: right\n      target_side: [left, right]\n",
    )
    _, _, prompt_mode, conditions = _load_eval_datasets(cfg)[0]
    assert prompt_mode == "task"
    assert conditions == {"pick_hand": "right", "target_side": ["left", "right"]}


def test_random_is_rejected_in_favour_of_an_explicit_list(tmp_path):
    root = make_root(tmp_path, "old", 5)
    cfg = _write_config(
        tmp_path,
        f"datasets:\n  - path: {root}\n    prompt: task\n"
        "    conditions:\n      pick_hand: random\n",
    )
    with pytest.raises(typer.BadParameter, match="cannot be 'random'"):
        _load_eval_datasets(cfg)


def test_overriding_a_condition_requires_prompt_task(tmp_path):
    # Otherwise the replayed prompt would describe a command we just replaced.
    root = make_root(tmp_path, "mirror", 5)
    cfg = _write_config(
        tmp_path,
        f"datasets:\n  - path: {root}\n    conditions:\n      target_side: right\n",
    )
    with pytest.raises(typer.BadParameter, match="prompt: task"):
        _load_eval_datasets(cfg)


def test_recorded_conditions_do_not_require_prompt_task(tmp_path):
    root = make_root(tmp_path, "mirror", 5)
    cfg = _write_config(
        tmp_path,
        f"datasets:\n  - path: {root}\n    conditions:\n      target_side: recorded\n",
    )
    assert _load_eval_datasets(cfg)[0][2] == "recorded"


def test_conditions_must_be_a_mapping(tmp_path):
    root = make_root(tmp_path, "old", 5)
    cfg = _write_config(tmp_path, f"datasets:\n  - path: {root}\n    conditions: [left]\n")
    with pytest.raises(typer.BadParameter, match="conditions must be a mapping"):
        _load_eval_datasets(cfg)
