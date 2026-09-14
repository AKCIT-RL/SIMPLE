"""
SIMPLE: SIMulation-based Policy Learning and Evaluation
"""

from __future__ import annotations
from simple.core.scene import Scene, TabletopScene
from simple.scenes.scene_manager import SceneManager
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simple.core import Asset

class WarehouseSuite(TabletopScene):
    name: str

    def __init__(self) -> None:
        self.uid = "warehouse:default"
        self.name = "warehouse"
        # Fallback path used only if Isaac Sim's Nucleus root can't be resolved
        # (see IsaacSimSimulator.__update_scene); normally the warehouse USD is
        # loaded relative to get_assets_root_path() instead.
        self.data_dir = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Environments/Simple_Warehouse/warehouse.usd"

    def set_table(self, table: Asset) -> None:
        self.table = table

    def set_table2(self, table2: Asset) -> None:
        self.table2 = table2

@SceneManager.register("warehouse")
class WarehouseSceneManager(SceneManager):

    def sample(self, exclude: list[str] | None = None) -> Scene:
        return WarehouseSuite()

    def load(self, scene_uid: str) -> Scene:
        return WarehouseSuite()
