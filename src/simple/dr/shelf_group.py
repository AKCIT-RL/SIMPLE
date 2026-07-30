"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

from simple.assets import AssetManager
from simple.core.asset import Asset
from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.types import Pose


@dataclass(frozen=True)
class ShelfSpec:
    """Real-world geometry of one shelf unit, calibrated against the staged
    `fixtures:corridor0` collision mesh (see
    docs/teleop_simple_study/toteweg_factory_scene_migration_plan.md, Phase 4)."""

    x_min: float
    x_max: float
    y: float  # fixed depth position for every tote on this shelf
    yaw_deg: float  # fixed yaw so the crate's front/recess mark faces the aisle
    tiers: Dict[str, float]  # tier letter -> tote resting Z (board top + stable-pose offset)


# Calibrated for estante_l1 / estante_l3 / estante_r1 (see migration plan doc,
# Phase 4 "Shelf-slot calibration" + "Row swap"). r2/r3 are not calibrated yet.
# Note: estante_l2 was deleted during the Phase 3 corridor correction (it
# overlapped estante_l1 by ~1.05m) -- the left row is only l1+l3.
#
# The whole l-block (l1+l3) and r-block (r1+r2+r3) were later swapped between
# Y-bands (each shelf also rotated 180 deg in place) per your request -- l1/l3
# now sit where r1/r2/r3 used to be and vice versa, with each block's
# near-table edge realigned to preserve that row's original table distance,
# and the 1.64m aisle gap preserved since the Y-bands themselves didn't move.
# Board Z-heights are untouched by this (pure XY translation + Z-axis
# rotation), re-verified directly against the transformed mesh regardless.
# yaw_deg flips 0<->180 versus the pre-swap table since each shelf's aisle
# side flipped with the Y-band swap (aisle is always "the other band").
SHELF_SPECS: Dict[str, ShelfSpec] = {
    "l1": ShelfSpec(
        x_min=-1.9961,
        x_max=-0.1768,
        y=1.6861,
        yaw_deg=180.0,
        tiers={"B": 0.5238, "D": 1.0077, "E": 1.2824},
    ),
    "l3": ShelfSpec(
        # l3 is a clone of l1's own geometry (Phase 2 "corridor completion
        # pass"), sharing the same y/z/scale/orientation -- only x differs.
        x_min=-3.8153,
        x_max=-1.9961,
        y=1.6861,
        yaw_deg=180.0,
        tiers={"B": 0.5238, "D": 1.0077, "E": 1.2824},
    ),
    "r1": ShelfSpec(
        x_min=0.1237,
        x_max=2.1414,
        y=-0.2985,
        yaw_deg=0.0,
        tiers={"B": 0.6326, "C": 0.9593, "D": 1.2861, "E": 1.6652},
    ),
}


def _yaw_quat(yaw_deg: float) -> List[float]:
    half = math.radians(yaw_deg) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _tier_bin_ranges(
    shelf: ShelfSpec, max_per_tier: int, tote_width: float, post_margin: float, min_gap: float
) -> List[Tuple[float, float]]:
    """Split the shelf's usable width into `max_per_tier` non-overlapping bins
    and return each bin's valid tote-center X range, inset so a tote placed
    anywhere in its own bin can never overlap a tote in a neighboring bin."""
    usable_x_min = shelf.x_min + post_margin
    usable_x_max = shelf.x_max - post_margin
    usable_width = usable_x_max - usable_x_min
    slot_width = usable_width / max_per_tier

    ranges = []
    for i in range(max_per_tier):
        bin_lo = usable_x_min + i * slot_width
        bin_hi = bin_lo + slot_width
        lo = bin_lo + tote_width / 2 + min_gap / 2
        hi = bin_hi - tote_width / 2 - min_gap / 2
        if lo > hi:
            raise ValueError(
                f"Tier bin too narrow for tote_width={tote_width} at max_per_tier={max_per_tier} "
                f"(slot_width={slot_width:.4f}); reduce max_per_tier or tote_width/min_gap."
            )
        ranges.append((lo, hi))
    return ranges


class ShelfGroupDR(Randomizer):
    """Spawns a stochastic population of the same asset across a set of named
    shelves: each shelf-tier independently holds 0..max_per_tier instances
    (resampled every reset), with each shelf's total rejection-resampled until
    it meets min_per_shelf. Non-overlapping X placement within a tier is
    guaranteed by construction (fixed bins, not rejection sampling) -- see the
    migration plan doc, Phase 4c.
    """

    cfg: "ShelfGroupDRCfg"

    def __init__(self, cfg: "ShelfGroupDRCfg") -> None:
        super().__init__(cfg)
        self.res_id, self.obj_id = cfg.asset_id.split(":")

    def _sample_shelf_occupancy(self, shelf: ShelfSpec) -> Dict[str, int]:
        tier_names = list(shelf.tiers.keys())
        while True:
            occupancy = {name: random.randint(0, self.cfg.max_per_tier) for name in tier_names}
            if sum(occupancy.values()) >= self.cfg.min_per_shelf:
                return occupancy

    def _placements_for_shelf(self, shelf_name: str) -> List[Tuple[str, str, Pose]]:
        shelf = SHELF_SPECS[shelf_name]
        bin_ranges = _tier_bin_ranges(
            shelf, self.cfg.max_per_tier, self.cfg.tote_width, self.cfg.post_margin, self.cfg.min_gap
        )
        occupancy = self._sample_shelf_occupancy(shelf)
        quat = _yaw_quat(shelf.yaw_deg)

        placements: List[Tuple[str, str, Pose]] = []
        for tier_name, count in occupancy.items():
            if count == 0:
                continue
            bin_indices = random.sample(range(self.cfg.max_per_tier), k=count)
            for bin_idx in bin_indices:
                lo, hi = bin_ranges[bin_idx]
                x = random.uniform(lo, hi)
                z = shelf.tiers[tier_name]
                pose = Pose(position=[x, shelf.y, z], quaternion=quat)
                placements.append((shelf_name, tier_name, pose))
        return placements

    def state_dict(self) -> Dict[str, Any]:
        if self._inner_state is None:
            return {}
        return {
            "res_id": self.res_id,
            "obj_id": self.obj_id,
            "placements": [
                {
                    "shelf": shelf_name,
                    "tier": tier_name,
                    "position": list(pose.position),
                    "quaternion": list(pose.quaternion),
                }
                for (_, shelf_name, tier_name, pose) in self._inner_state
            ],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict:
            self._inner_state = None
            return
        asset_manager = AssetManager.get(state_dict["res_id"])
        self._inner_state = [
            (
                asset_manager.load(state_dict["obj_id"]),
                p["shelf"],
                p["tier"],
                Pose(position=p["position"], quaternion=p["quaternion"]),
            )
            for p in state_dict["placements"]
        ]

    def __call__(self, split: str, **kwargs) -> List[Tuple[Asset, Pose]]:
        asset_manager = AssetManager.get(self.res_id)
        instances: List[Tuple[Asset, str, str, Pose]] = []
        for shelf_name in self.cfg.shelves:
            for shelf_name_out, tier_name, pose in self._placements_for_shelf(shelf_name):
                instances.append((asset_manager.load(self.obj_id), shelf_name_out, tier_name, pose))
        instances = super()._transient(instances)
        return [(asset, pose) for (asset, _shelf, _tier, pose) in instances]


@dataclass
class ShelfGroupDRCfg(RandomizerCfg):
    asset_id: str | None = None  # e.g., "totes:toteweg"
    shelves: List[str] = field(default_factory=lambda: ["l1", "l3", "r1"])
    max_per_tier: int = 3
    min_per_shelf: int = 2
    tote_width: float = 0.197  # toteweg lateral footprint, from Phase 1 calibration
    post_margin: float = 0.15  # clearance from shelf frame legs at each tier's ends (placeholder)
    min_gap: float = 0.03  # minimum clearance between neighboring totes in the same tier
    randmizer_class: Type[Randomizer] = ShelfGroupDR

    def max_total_count(self) -> int:
        """Upper bound on how many instances a single reset can ever produce
        across all configured shelves -- used by recording tooling to size a
        fixed (padded) `observation.object_poses` schema up front, since the
        real per-reset count varies (see migration plan doc, Phase 4c)."""
        return sum(len(SHELF_SPECS[shelf].tiers) * self.max_per_tier for shelf in self.shelves)
