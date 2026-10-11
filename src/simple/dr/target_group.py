"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Type

from simple.assets import AssetManager
from simple.core.asset import Asset
from simple.core.randomizer import Randomizer, RandomizerCfg


class TargetGroupDR(Randomizer):
    """Spawns N pickable instances of the *same* asset_id.

    Unlike DistractorDR, this does not exclude previously-picked uids
    between draws -- duplicates are the whole point, since all instances
    are the same object (e.g. 1-2 identical totes). Returns an ordered
    list rather than a uid-keyed dict, so duplicate uids never collide.
    """

    cfg: "TargetGroupDRCfg"

    def __init__(self, cfg: "TargetGroupDRCfg") -> None:
        super().__init__(cfg)
        self.res_id, self.obj_id = cfg.asset_id.split(":")

    def state_dict(self) -> Dict[str, Any]:
        if self._inner_state is None:
            return {}
        return {
            "res_id": self.res_id,
            "obj_id": self.obj_id,
            "count": len(self._inner_state),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict:
            self._inner_state = None
            return
        count = state_dict["count"]
        asset_manager = AssetManager.get(state_dict["res_id"])
        self._inner_state = [asset_manager.load(state_dict["obj_id"]) for _ in range(count)]

    def _sample_count(self) -> int:
        count = self.cfg.count
        if isinstance(count, (list, tuple)):
            assert len(count) > 0, "count choices must not be empty"
            return int(random.choice(count))
        return int(count)

    def __call__(self, split: str, **kwargs) -> List[Asset]:
        count = self._sample_count()
        asset_manager = AssetManager.get(self.res_id)
        assets = [asset_manager.load(self.obj_id) for _ in range(count)]
        return super()._transient(assets)


@dataclass
class TargetGroupDRCfg(RandomizerCfg):
    asset_id: str | None = None  # e.g., "res_id:obj_id"
    # Fixed instance count, or a list of choices sampled once per episode
    # (e.g. [1, 2] to alternate between spawning 1 or 2 instances).
    count: int | List[int] = 1
    randmizer_class: Type[Randomizer] = TargetGroupDR
