"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Any

import numpy as np

from .asset_manager import AssetManager
from simple.core.asset import Asset
from simple.core.object import SemanticAnnotated, SpatialAnnotated

Totes_Names = {
    0: "bin_b04",
    1: "bin_b04_red",
    2: "bin_b04_blue",
}

# Color variants reuse the bin_b04 meshes/USD on disk but carry a distinct
# uid/label (so multiple totes can coexist in one MuJoCo scene — the engine
# names bodies by asset.label, and duplicates fail to compile) and a fixed RGBA
# tint the MuJoCo engine applies to the object geoms (see _build_object).
# NOTE: the tint is MuJoCo-side only; the Isaac Sim replay renders the bin_b04
# USD's own material (color override there is a render-stage concern).
Totes_Variants: dict[str, dict[str, Any]] = {
    "bin_b04_red": {"base": "bin_b04", "rgba": [0.90, 0.02, 0.02, 1.0]},
    "bin_b04_blue": {"base": "bin_b04", "rgba": [0.02, 0.08, 0.90, 1.0]},
}


def _estimate_stable_z_from_obj(obj_path: str, default: float = 0.12) -> float:
    """Estimate a reasonable stable pose z-offset from the visual mesh."""
    z_values: list[float] = []
    try:
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    z_values.append(float(parts[3]))
    except Exception:
        return default

    if not z_values:
        return default
    
    # Do not force a default height if the geometric min is perfectly flush (0.0).
    return -min(z_values)


class TotesAsset(Asset, SemanticAnnotated, SpatialAnnotated):
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
        rgba: list[float] | None = None,
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
        # Optional MuJoCo geom tint; the engine's _build_object reads asset.rgba
        # when present (color variants like bin_b04_red set this).
        self.rgba = rgba
        self.stable_poses = (
            stable_poses if stable_poses is not None else np.array([[0, 0, 0.12, 1, 0, 0, 0]])
        )
        # Keep these for compatibility with SpatialAnnotated checks.
        self.canonical_grasps = []
        self.functional_grasps = {}
        self.keypoints = {}
        self.axes = {}

    def __repr__(self) -> str:
        return f"TotesAsset(uid={self.uid}, name={self.name})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "res_id": "totes",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "usd_path": self.usd_path,
            "description": self.description,
        }


@AssetManager.register("totes")
class TotesAssetManager(AssetManager):
    def __init__(self) -> None:
        self.src_dir = os.path.join(os.path.dirname(__file__), "totes")

    def _resolve_name(self, asset_id: str) -> str:
        if asset_id.isdigit():
            idx = int(asset_id)
            assert idx in Totes_Names, f"Invalid totes asset ID: {asset_id}"
            return Totes_Names[idx]
        assert asset_id in Totes_Names.values(), f"Invalid totes asset ID: {asset_id}"
        return asset_id

    def load(self, asset_id: str) -> Asset:
        name = self._resolve_name(asset_id)
        # Color variants resolve to their base folder on disk (same meshes/USD);
        # only uid/label and the rgba tint differ.
        variant = Totes_Variants.get(name)
        folder = variant["base"] if variant is not None else name
        base_dir = os.path.join(self.src_dir, folder)
        assert os.path.isdir(base_dir), f"Totes asset folder not found: {base_dir}"

        collision_dir = os.path.join(base_dir, "MJCF", "collision")
        collision_meshes_mujoco = sorted(glob.glob(os.path.join(collision_dir, "*.obj")))
        assert collision_meshes_mujoco, f"No collision meshes found at: {collision_dir}"

        visual_mesh = os.path.join(base_dir, "MJCF", "visuals", "Bin_B04_01.obj")
        assert os.path.exists(visual_mesh), f"Visual mesh not found: {visual_mesh}"

        # All variants share the base USD: the colour is NOT in the asset. Both
        # engines take it from `rgba` below — MuJoCo as a geom tint, Isaac by
        # binding an OmniPBR material (see IsaacSimSimulator._bind_color_material;
        # colouring the bin's own material has no effect on the render).
        usd_path = os.path.join(base_dir, f"{folder}.usd")
        assert os.path.exists(usd_path), f"USD file not found: {usd_path}"

        stable_z = _estimate_stable_z_from_obj(visual_mesh, default=0.12)
        stable_poses = np.array([[0.0, 0.0, stable_z, 1.0, 0.0, 0.0, 0.0]])

        return TotesAsset(
            uid=name,
            label=name,
            name=name,
            usd_path=usd_path,
            collision_mesh_curobo=visual_mesh,
            collision_meshes_mujoco=collision_meshes_mujoco,
            description="NVIDIA SimReady tote bin asset",
            stable_poses=stable_poses,
            rgba=variant["rgba"] if variant is not None else None,
        )

    def sample(self, exclude: list[str] | None = None) -> Asset:
        exclude_set = set(str(e) for e in (exclude or []))
        # Random sampling only draws base totes; color variants are task-specific
        # and must be requested explicitly by name.
        candidates = [
            name
            for name in Totes_Names.values()
            if name not in exclude_set and name not in Totes_Variants
        ]
        if not candidates:
            raise ValueError("No tote assets available after exclusions.")
        return self.load(random.choice(candidates))

    def __len__(self) -> int:
        return len(Totes_Names)

