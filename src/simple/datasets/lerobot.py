"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
import json
from pathlib import Path


def get_episode_lerobot(dataset, eps_idx, data_format=None, load_frames=True):
    """Return (environment_config, frames) for one episode.

    `load_frames=False` skips materializing the frame list entirely. Building it
    decodes one video frame per step, which dominates the cost on consolidated
    (psi0-format) datasets -- their episodes are whole demos, thousands of frames
    long -- and the result is only consumed by the `vlt` baseline (see
    eval_decoupled_wbc.py's reset_kwargs). Every other policy throws it away.
    """
    def _to_int(value):
        item = getattr(value, "item", None)
        if callable(item):
            return int(item())
        return int(value)

    episode = None
    if load_frames:
        from_idx = _to_int(dataset.episode_data_index["from"][eps_idx])
        to_idx = _to_int(dataset.episode_data_index["to"][eps_idx])
        episode = [dataset[i] for i in range(from_idx, to_idx)]

    env_conf = json.loads(dataset.meta.episodes[eps_idx]['environment_config'])
    if "scene" in env_conf.get("dr_state_dict", {}):  # FIXME
        env_conf["dr_state_dict"]["scene"]["uid"] = env_conf["dr_state_dict"]["scene"]["uid"].replace("102344280", "scene3")
    # import pickle; pickle.dump(env_conf, open(f"env_conf_{eps_idx}.pkl", "wb"))
    # import pickle; env_conf = pickle.loads(open(f"env_conf_{eps_idx}.pkl", "rb").read())
    return env_conf, episode


def _load_episode_tasks(data_dir) -> dict[int, str]:
    """Load {task_index: prompt} from meta/tasks.jsonl, {} if absent."""
    meta_file = Path(data_dir) / "meta" / "tasks.jsonl"
    if not meta_file.exists():
        return {}

    tasks: dict[int, str] = {}
    with open(meta_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            tasks[int(entry["task_index"])] = entry["task"]
    return tasks


def _load_episode_prompts(data_dir) -> dict[int, str]:
    """Load the recorded language prompt of each episode from episodes.jsonl.

    Tasks whose prompt varies per episode (sampled quantities, targets) must be
    replayed with the prompt that was actually recorded; keying off tasks.jsonl
    alone would collapse every episode onto one instruction.

    The `tasks` field carries the prompt itself in raw/rendered captures, but a
    *task index* in the consolidated psi0 format (postprocess_psi0_sonic.py
    reassigns indices while merging sessions). Indices are resolved through
    tasks.jsonl so both layouts yield a plain string -- handing the raw index to
    a policy as its instruction would be silent nonsense.
    """
    meta_file = Path(data_dir) / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return {}

    tasks_by_index: dict[int, str] | None = None
    prompts: dict[int, str] = {}
    with open(meta_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            ep_idx = entry.get("episode_index", None)
            ep_tasks = entry.get("tasks", None)
            if ep_idx is None or not ep_tasks:
                continue
            value = ep_tasks[0]
            if isinstance(value, str):
                prompts[int(ep_idx)] = value
                continue
            if tasks_by_index is None:
                tasks_by_index = _load_episode_tasks(data_dir)
            resolved = tasks_by_index.get(int(value))
            if resolved is not None:
                prompts[int(ep_idx)] = resolved
    return prompts
