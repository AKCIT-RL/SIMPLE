"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Simple colored cube assets for primitive manipulation tasks (e.g. the arm-only
colour-box sorting task). Every variant shares ONE cube mesh/USD on disk
(cube.obj + cube.usda, a 6 cm cube); only uid/label and the RGBA colour differ
-- the engines take the colour from ``asset.rgba`` (MuJoCo as a geom tint, Isaac
by binding an OmniPBR material), exactly like the totes colour variants. The
distinct uid/label lets several boxes coexist in one MuJoCo scene (the engine
names bodies by ``asset.label``, and duplicate names fail to compile).
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

from .asset_manager import AssetManager
from simple.core.asset import Asset
from simple.core.object import SemanticAnnotated, SpatialAnnotated

# 6 cm cube -> half-extent 0.03 m; the mesh is centered on the origin, so a box
# rests with its center 0.03 m above the surface it stands on.
_CUBE_HALF_EXTENT = 0.03

# uid -> RGBA. The colour names double as the language cue in the task prompt.
Boxes_Variants: dict[str, dict[str, Any]] = {
    "box_red": {"rgba": [0.90, 0.02, 0.02, 1.0]},
    "box_green": {"rgba": [0.02, 0.70, 0.05, 1.0]},
    "box_blue": {"rgba": [0.02, 0.08, 0.90, 1.0]},
}


class BoxAsset(Asset, SemanticAnnotated, SpatialAnnotated):
    def __init__(
        self,
        uid: str,
        label: str,
        name: str,
        usd_path: str,
        collision_mesh_curobo: str,
        collision_meshes_mujoco: list[str],
        rgba: list[float],
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
        self.rgba = rgba
        # Rests flat: center one half-extent above the surface, identity rotation.
        self.stable_poses = np.array(
            [[0.0, 0.0, _CUBE_HALF_EXTENT, 1.0, 0.0, 0.0, 0.0]]
        )
        # Compatibility with SpatialAnnotated checks (no grasp annotations here).
        self.canonical_grasps = []
        self.functional_grasps = {}
        self.keypoints = {}
        self.axes = {}

    def __repr__(self) -> str:
        return f"BoxAsset(uid={self.uid}, name={self.name})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "res_id": "boxes",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "usd_path": self.usd_path,
            "description": self.description,
        }


@AssetManager.register("boxes")
class BoxesAssetManager(AssetManager):
    def __init__(self) -> None:
        self.src_dir = os.path.join(os.path.dirname(__file__), "boxes")

    def load(self, asset_id: str) -> Asset:
        assert asset_id in Boxes_Variants, (
            f"Unknown box asset_id: {asset_id!r} (have {list(Boxes_Variants)})"
        )
        cube_obj = os.path.join(self.src_dir, "cube.obj")
        cube_usd = os.path.join(self.src_dir, "cube.usda")
        assert os.path.exists(cube_obj), f"Cube mesh not found: {cube_obj}"
        assert os.path.exists(cube_usd), f"Cube USD not found: {cube_usd}"
        return BoxAsset(
            uid=asset_id,
            label=asset_id,
            name=asset_id,
            usd_path=cube_usd,
            collision_mesh_curobo=cube_obj,
            collision_meshes_mujoco=[cube_obj],
            rgba=list(Boxes_Variants[asset_id]["rgba"]),
            description="Solid-colour 6 cm cube for primitive manipulation tasks.",
        )

    def sample(self, exclude: list[str] | None = None) -> Asset:
        exclude_set = set(exclude or [])
        candidates = [name for name in Boxes_Variants if name not in exclude_set]
        if not candidates:
            raise ValueError("No box assets available after exclusions.")
        return self.load(random.choice(candidates))

    def __len__(self) -> int:
        return len(Boxes_Variants)
