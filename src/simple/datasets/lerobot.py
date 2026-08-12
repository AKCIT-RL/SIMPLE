"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""
from collections.abc import Sequence
import json


class LazyEpisode(Sequence):
    """A read-only view over one episode's frames, decoded on access.

    Behaves like the list it replaces (``len``, indexing, negative indices,
    slicing, iteration), so callers such as ``EpisodeExtractor`` need no change.

    Why this is not a plain list: ``LeRobotDataset.__getitem__`` decodes video
    frames, so materialising a whole episode up front costs one full decode of
    the episode's video before the rollout even starts. The closed-loop eval
    path only reads ``env_conf`` and never touches these frames, so that work
    was always discarded.

    For the short eval datasets (``simple-eval`` episodes hold a single row)
    this was invisible. It stops being invisible when a full teleop dataset is
    used as the source of initial conditions: episodes of several thousand
    frames take minutes to decode, allocate tens of GB, and -- with Isaac Sim
    already loaded in the same process -- abort in libswscale with
    ``BlockingIOError: [Errno 11]`` while initialising its scaling graph.

    Replay, which does read every frame, decodes exactly as much as before,
    just spread across the rollout instead of paid in one burst.
    """

    __slots__ = ("_dataset", "_from_idx", "_to_idx")

    def __init__(self, dataset, from_idx: int, to_idx: int):
        self._dataset = dataset
        self._from_idx = from_idx
        self._to_idx = to_idx

    def __len__(self) -> int:
        return self._to_idx - self._from_idx

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            # Required, not defensive: without it iteration never terminates.
            raise IndexError(f"episode index out of range: {index}")
        return self._dataset[self._from_idx + index]

    def __repr__(self) -> str:
        return f"LazyEpisode(frames={len(self)}, offset={self._from_idx})"


def get_episode_lerobot(dataset, eps_idx, data_format=None):
    def _to_int(value):
        item = getattr(value, "item", None)
        if callable(item):
            return int(item())
        return int(value)

    from_idx = _to_int(dataset.episode_data_index["from"][eps_idx])
    to_idx = _to_int(dataset.episode_data_index["to"][eps_idx])
    episode = LazyEpisode(dataset, from_idx, to_idx)

    env_conf = json.loads(dataset.meta.episodes[eps_idx]['environment_config'])
    # FIXME: household (hssd-scene) tasks record a raw collection-time scene id and this
    # remaps it to the fixed scene available in eval. Tasks with a different DR family
    # (e.g. shelf_group-based warehouse tasks) have no "scene" entry in dr_state_dict at
    # all, so this must be a no-op for them rather than a KeyError.
    scene_cfg = env_conf.get("dr_state_dict", {}).get("scene")
    if scene_cfg is not None and "uid" in scene_cfg:
        scene_cfg["uid"] = scene_cfg["uid"].replace("102344280", "scene3")
    # import pickle; pickle.dump(env_conf, open(f"env_conf_{eps_idx}.pkl", "wb"))
    # import pickle; env_conf = pickle.loads(open(f"env_conf_{eps_idx}.pkl", "rb").read())
    return env_conf, episode
