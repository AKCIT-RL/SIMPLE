"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Industrial graspable parts (gears, t-connectors, ...) extracted from NVIDIA
Isaac Props/Factory USD assets into MuJoCo-ready dynamic objects.

Asset folders live under `data/assets/industrial_parts/<name>/` and are produced
by `scripts/industrial/extract_part_mesh.py` (USD -> visual OBJ + VHACD convex
hulls + stable poses). Same on-disk contract as the graspnet / totes managers, so
the MuJoCo engine builds the free rigid body from `collision_meshes_mujoco` and
Isaac Sim renders `usd_path` at replay time. Keeping `uid == label == name`
avoids the replay name-mismatch class of bug (see INDUSTRIAL_ADAPTATION_CHANGES).
"""

from __future__ import annotations

import glob
import os
import random
from typing import Any

import numpy as np

from simple.core.asset import Asset
from simple.core.object import SemanticAnnotated, SpatialAnnotated
from simple.utils import resolve_data_path

from .asset_manager import AssetManager

Industrial_Part_Names = {
    0: "factory_gear_large",
    1: "t_connector_physics",
}

_REL_DIR = "assets/industrial_parts"


class IndustrialPartAsset(Asset, SemanticAnnotated, SpatialAnnotated):
    def __init__(
        self,
        uid: str,
        label: str,
        name: str,
        usd_path: str,
        collision_mesh_curobo: str,
        collision_meshes_mujoco: list[str],
        description: str | None = None,
        stable_poses: np.ndarray | None = None,
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
        self.stable_poses = (
            stable_poses
            if stable_poses is not None
            else np.array([[0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]])
        )
        # Stubs kept for SpatialAnnotated/SemanticAnnotated compatibility (mirrors totes).
        self.canonical_grasps = []
        self.functional_grasps = {}
        self.keypoints = {}
        self.axes = {}

    def __repr__(self) -> str:
        return f"IndustrialPartAsset(uid={self.uid}, name={self.name})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "res_id": "industrial_parts",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "usd_path": self.usd_path,
            "description": self.description,
        }


@AssetManager.register("industrial_parts")
class IndustrialPartsAssetManager(AssetManager):
    def _resolve_name(self, asset_id: str) -> str:
        if str(asset_id).isdigit():
            idx = int(asset_id)
            assert idx in Industrial_Part_Names, f"Invalid industrial part ID: {asset_id}"
            return Industrial_Part_Names[idx]
        assert (
            asset_id in Industrial_Part_Names.values()
        ), f"Invalid industrial part ID: {asset_id}"
        return asset_id

    def load(self, asset_id: str) -> Asset:
        name = self._resolve_name(asset_id)
        base_dir = resolve_data_path(os.path.join(_REL_DIR, name))
        assert os.path.isdir(base_dir), f"Industrial part folder not found: {base_dir}"

        collision_dir = os.path.join(base_dir, "collision")
        collision_meshes_mujoco = sorted(glob.glob(os.path.join(collision_dir, "convex_piece_*.obj")))
        assert collision_meshes_mujoco, f"No collision meshes found at: {collision_dir}"

        visual_mesh = os.path.join(base_dir, "visual.obj")
        assert os.path.exists(visual_mesh), f"Visual mesh not found: {visual_mesh}"

        usd_path = os.path.join(base_dir, f"{name}.usd")
        assert os.path.exists(usd_path), f"USD file not found: {usd_path}"

        stable_path = os.path.join(base_dir, "stable_poses.npy")
        stable_poses = (
            np.load(stable_path, allow_pickle=True)
            if os.path.exists(stable_path)
            else None
        )

        return IndustrialPartAsset(
            uid=name,
            label=name,
            name=name,
            usd_path=usd_path,
            collision_mesh_curobo=visual_mesh,
            collision_meshes_mujoco=collision_meshes_mujoco,
            description="Industrial part extracted from NVIDIA Isaac USD",
            stable_poses=stable_poses,
        )

    def sample(self, exclude: list[str] | None = None) -> Asset:
        exclude_set = set(str(e) for e in (exclude or []))
        candidates = [
            name for name in Industrial_Part_Names.values() if name not in exclude_set
        ]
        if not candidates:
            raise ValueError("No industrial part assets available after exclusions.")
        return self.load(random.choice(candidates))

    def __len__(self) -> int:
        return len(Industrial_Part_Names)
