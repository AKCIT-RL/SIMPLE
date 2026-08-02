"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from dataclasses import dataclass
from typing import Any

class Asset:

    uid: str
    # name: str
    usd_path: str # TODO move to sub asset classes
    collision_mesh_curobo: str 
    collision_meshes_mujoco: list[str] 

    def __init__(
        self, 
        uid: str, # name: str, 
        usd_path: str,
        collision_mesh_curobo: str,
        collision_meshes_mujoco: list[str]
    ) -> None:
        self.uid = uid
        # self.name = name
        self.usd_path = usd_path
        self.collision_mesh_curobo = collision_mesh_curobo
        self.collision_meshes_mujoco = collision_meshes_mujoco
    
    # def to_dict(self) -> dict[str, Any]:
    #     pass

class ArticulatedAsset:
    uid: str
    # name: str
    usd_path: str   | None
    mjcf_path: str
    # Static props (warehouse furniture) ride the articulated path because that's
    # how MuJoCo attaches a whole MJCF at a frame, but they have ZERO joints.
    # Isaac Sim must render them as a plain referenced prim + XForm pose instead
    # of a SingleArticulation (which requires an actual articulation).
    static: bool
    # Scale to apply to the USD when referenced into a metres stage. NVIDIA
    # warehouse assets are authored in centimetres (metersPerUnit = 0.01);
    # add_reference_to_stage does NOT convert units (only Omniverse's metrics
    # assembler does, via an `unitsResolve` xformOp), so without this they come
    # in 100x too large.
    usd_scale: float

    def __init__(
        self,
        uid: str,
        usd_path: str | None,
        mjcf_path: str,
        static: bool = False,
        usd_scale: float = 1.0,
    ) -> None:
        self.uid = uid
        self.usd_path = usd_path
        self.mjcf_path = mjcf_path
        self.static = static
        self.usd_scale = usd_scale


