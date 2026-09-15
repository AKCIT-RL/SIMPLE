"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import glob
import os
from typing import Any

from .asset_manager import AssetManager
from simple.core.asset import Asset
from simple.core.object import SemanticAnnotated

Fixtures_Names = {
    0: "corridor0",
}


class FixtureAsset(Asset, SemanticAnnotated):
    """A static, non-manipulable scenario prop with real MuJoCo collision
    (e.g. a shelf aisle) -- built via StaticObjectActor/_build_static_object,
    never a free-floating rigid body, and never routed through SpatialDR
    (its pose is authored/fixed by the task, not domain-randomized like
    target/container/distractor roles).
    """

    def __init__(
        self,
        uid: str,
        label: str,
        name: str,
        usd_path: str,
        collision_mesh_curobo: str,
        collision_meshes_mujoco: list[str],
        description: str | None = None,
    ) -> None:
        super().__init__(
            uid=uid,
            usd_path=usd_path,
            collision_mesh_curobo=collision_mesh_curobo,
            collision_meshes_mujoco=collision_meshes_mujoco,
        )
        self.label = label
        self.name = name
        self.description = description

    def __repr__(self) -> str:
        return f"FixtureAsset(uid={self.uid}, name={self.name})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "res_id": "fixtures",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "usd_path": self.usd_path,
            "description": self.description,
        }


@AssetManager.register("fixtures")
class FixturesAssetManager(AssetManager):
    def __init__(self) -> None:
        self.src_dir = os.path.join(os.path.dirname(__file__), "fixtures")

    def _resolve_name(self, asset_id: str) -> str:
        if asset_id.isdigit():
            idx = int(asset_id)
            assert idx in Fixtures_Names, f"Invalid fixtures asset ID: {asset_id}"
            return Fixtures_Names[idx]
        assert asset_id in Fixtures_Names.values(), f"Invalid fixtures asset ID: {asset_id}"
        return asset_id

    def load(self, asset_id: str) -> Asset:
        name = self._resolve_name(asset_id)
        base_dir = os.path.join(self.src_dir, name)
        assert os.path.isdir(base_dir), f"Fixtures asset folder not found: {base_dir}"

        collision_dir = os.path.join(base_dir, "MJCF", "collision")
        collision_meshes_mujoco = sorted(glob.glob(os.path.join(collision_dir, "*.obj")))
        assert collision_meshes_mujoco, f"No collision meshes found at: {collision_dir}"

        visuals_dir = os.path.join(base_dir, "MJCF", "visuals")
        found_visuals = sorted(glob.glob(os.path.join(visuals_dir, "*.obj")))
        assert found_visuals, f"No visual mesh found under: {visuals_dir}"
        visual_mesh = found_visuals[0]

        usd_path = os.path.join(base_dir, f"{name}.usd")
        assert os.path.exists(usd_path), f"USD file not found: {usd_path}"

        return FixtureAsset(
            uid=name,
            label=name,
            name=name,
            usd_path=usd_path,
            collision_mesh_curobo=visual_mesh,
            collision_meshes_mujoco=collision_meshes_mujoco,
            description="Static scenario fixture (fixed collision geometry, no free joint)",
        )

    def sample(self, exclude: list[str] | None = None) -> Asset:
        exclude_set = set(str(e) for e in (exclude or []))
        candidates = [name for name in Fixtures_Names.values() if name not in exclude_set]
        if not candidates:
            raise ValueError("No fixtures assets available after exclusions.")
        import random
        return self.load(random.choice(candidates))

    def __len__(self) -> int:
        return len(Fixtures_Names)
