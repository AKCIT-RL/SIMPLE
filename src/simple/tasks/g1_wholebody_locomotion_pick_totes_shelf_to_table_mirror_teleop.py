"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Mirror-table variant of the shelf-to-table teleop task: two mirrored delivery
tables (left / right), and the episode's language prompt commands BOTH which
hand to pick with (left / right / both) and which table to deliver to
(left / right). The pick hand only conditions the prompt/behaviour -- it is NOT
scored; success is strict on the delivery side (target tote resting upright on
the commanded table, released from the hand). Delivering to the wrong table is
a failure -- that is the language-generalization signal the eval measures.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import transforms3d as t3d

from simple.assets import AssetManager
from simple.dr import LanguageDRCfg
from simple.tasks.registry import TaskRegistry
from simple.tasks.g1_wholebody_locomotion_pick_totes_shelf_to_table_teleop import (
    G1WholebodyLocomotionPickTotesShelfToTableTaskTeleop,
    _ROBOT_SPAWN_POSITION,
    _TABLE_HEIGHT,
    _TABLE_POSITION_XY,
    _TABLE_ROTATION_Z,
    _TABLE_SIZE,
)

# The corridor runs along X; its two open ends (past the shelves / r-boxes) are
# the robot's LEFT and RIGHT when it faces the shelves at spawn. The existing
# table sits past the -X end -> that is the LEFT table (kept unchanged). The
# RIGHT table is its mirror IN X across the robot's left/right symmetry plane
# (x = _CORRIDOR_CENTER_X, defaulting to the robot's spawn X), landing
# symmetrically past the +X end.
#
# NB: mirroring in Y (a first attempt) put both tables at the same -X end,
# overlapping -- the corridor's left/right axis is X, so the mirror is in X.
# Same lateral offset (y) and orientation for both. Calibrate _CORRIDOR_CENTER_X
# if the right table doesn't land cleanly past the +X opening.
_CORRIDOR_CENTER_X = _ROBOT_SPAWN_POSITION[0]

# Prompt slot fillers. NB: "both" reads "with both hands" (no "your"), so the
# possessive lives inside the per-hand phrase rather than the template.
_HAND_CLAUSE = {
    "left": "your left hand",
    "right": "your right hand",
    "both": "both hands",
}
_PICK_HANDS = ("left", "right", "both")
_TABLE_SIDES = ("left", "right")


@TaskRegistry.register("g1_wholebody_locomotion_pick_totes_shelf_to_table_mirror_teleop")
class G1WholebodyLocomotionPickTotesShelfToTableMirrorTaskTeleop(
    G1WholebodyLocomotionPickTotesShelfToTableTaskTeleop
):
    uid: str = "g1_wholebody_locomotion_pick_totes_shelf_to_table_mirror_teleop"
    label: str = "G1 TELEOP Pick Totes Shelf to Left/Right Table"
    description: str = (
        "Pick the blue tote from the estante_l1/estante_l3 shelves with the "
        "commanded hand and deliver it to the commanded (left or right) table. "
        "Two mirrored tables; the pick hand and delivery side are set per "
        "episode and encoded in the language prompt."
    )

    # Same DR as the base task, but a two-slot language template: {hand_clause}
    # (which hand) and {side} (which table). Every other randomizer (spatial,
    # shelf_group, camera, lighting, material) is shared with the base task.
    dr_cfgs = dict(
        G1WholebodyLocomotionPickTotesShelfToTableTaskTeleop.dr_cfgs,
        language=LanguageDRCfg(
            instructions=[
                "Pick up the blue tote from the shelf with {hand_clause} and place it on the {side} table.",
            ]
        ),
    )

    def __init__(
        self,
        *args,
        pick_hand: str = "random",
        target_side: str = "random",
        **kwargs,
    ) -> None:
        # Captured here (not in **kwargs) so they don't leak downstream into
        # RobotRegistry.make / Task.__init__. "random" (or any unknown value)
        # means sample per episode; "left"/"right"/"both" pins it for
        # controlled data collection.
        self._fixed_pick_hand = pick_hand
        self._fixed_target_side = target_side
        self._pick_hand: Optional[str] = None
        self._target_side: Optional[str] = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Per-episode conditions (pick hand + delivery side)
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_or_fixed(fixed: str, choices: tuple[str, ...]) -> str:
        return fixed if fixed in choices else random.choice(choices)

    def _resolve_episode_conditions(self, options: Optional[dict[str, Any]]) -> None:
        """Set _pick_hand / _target_side for this episode. On replay reuse the
        recorded values (so a re-rendered capture keeps the same prompt and
        delivery target); otherwise honour the fixed CLI/env override, else
        sample uniformly."""
        state = (options or {}).get("state_dict") if options else None
        rec_hand = state.get("pick_hand") if isinstance(state, dict) else None
        rec_side = state.get("target_side") if isinstance(state, dict) else None
        self._pick_hand = rec_hand or self._sample_or_fixed(self._fixed_pick_hand, _PICK_HANDS)
        self._target_side = rec_side or self._sample_or_fixed(self._fixed_target_side, _TABLE_SIDES)

    @property
    def pick_hand(self) -> str:
        assert self._pick_hand is not None, "call reset() first"
        return self._pick_hand

    @property
    def target_side(self) -> str:
        assert self._target_side is not None, "call reset() first"
        return self._target_side

    def reset(self, seed: int | None = None, options: Optional[dict[str, Any]] = None) -> None:
        # Must run BEFORE super().reset(), which calls _build_instruction()
        # (below) -- that needs _pick_hand / _target_side already set.
        self._resolve_episode_conditions(options)
        super().reset(seed, options)

    # ------------------------------------------------------------------
    # Overridden hooks
    # ------------------------------------------------------------------
    def _add_delivery_tables(self) -> None:
        """Two mirrored tables: `table_left` at the existing position, and
        `table_right` reflected across the corridor centerline. Distinct,
        non-substring-colliding names so the contact scan can tell them apart
        (see tote_location_counts)."""
        z = _TABLE_HEIGHT - 0.5 * _TABLE_SIZE[2]
        quat = t3d.euler.euler2quat(0, 0, _TABLE_ROTATION_Z).tolist()
        y = _TABLE_POSITION_XY[1]
        left_x = _TABLE_POSITION_XY[0]
        right_x = 2.0 * _CORRIDOR_CENTER_X - left_x
        for name, x in (("table_left", left_x), ("table_right", right_x)):
            asset = AssetManager.create(
                "primitive:box",
                size=list(_TABLE_SIZE),
                position=[x, y, z],
                quaternion=quat,
            )
            self._layout.add_primitive(name, asset)

    def _build_instruction(self, split: str) -> str:
        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        template = lang_dr(split)  # raw template with {hand_clause}/{side} slots
        return template.format(
            hand_clause=_HAND_CLAUSE[self._pick_hand],
            side=self._target_side,
        )

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update({"pick_hand": self._pick_hand, "target_side": self._target_side})
        return sd

    # ------------------------------------------------------------------
    # Side-aware contact classification + strict success
    # ------------------------------------------------------------------
    def tote_location_counts(self, *args, keys: Optional[list[str]] = None, **kwargs) -> Dict[str, int]:
        """Same single contact-scan as the base task, but classifying the two
        delivery tables separately (`table_left` / `table_right`). Keeps a
        combined `table` count for backward compatibility with any consumer
        that read the base task's key."""
        mujoco_env = kwargs.get("mujoco_env", None)
        counts = {"shelf": 0, "table": 0, "table_left": 0, "table_right": 0, "ground": 0}
        live_keys = self._live_target_keys() if keys is None else list(keys)
        if not live_keys or mujoco_env is None:
            return counts

        body_name_by_key = self._tote_body_names(live_keys)
        live_body_names = set(body_name_by_key.values())

        mj_data = mujoco_env.mjData
        mj_model = mujoco_env.mjModel

        tote_contacts: Dict[str, set] = {name: set() for name in live_body_names}
        for i_contact in range(mj_data.ncon):
            contact = mj_data.contact[i_contact]
            body1 = mj_model.body(mj_model.geom(contact.geom1).bodyid).name
            body2 = mj_model.body(mj_model.geom(contact.geom2).bodyid).name
            for tote_name in live_body_names:
                if tote_name in body1:
                    tote_contacts[tote_name].add(body2)
                if tote_name in body2:
                    tote_contacts[tote_name].add(body1)

        for key, body_name in body_name_by_key.items():
            others = tote_contacts[body_name]
            if any("table_left" in o for o in others):
                if self._tote_is_upright(key):
                    counts["table_left"] += 1
                    counts["table"] += 1
            elif any("table_right" in o for o in others):
                if self._tote_is_upright(key):
                    counts["table_right"] += 1
                    counts["table"] += 1
            elif any("corridor0" in o for o in others):
                counts["shelf"] += 1
            elif any(o == "ground" for o in others):
                counts["ground"] += 1
        return counts

    def check_target_tote_on_table(self, *args, **kwargs) -> bool:
        """True only if the target tote is resting upright on the table matching
        the COMMANDED side. Delivering to the other table does not count -- this
        is the strict, side-specific success the base task's compute_reward
        already builds on."""
        counts = self.target_tote_location_counts(*args, **kwargs)
        return counts.get(f"table_{self._target_side}", 0) > 0

    def delivered_to_wrong_table(self, *args, **kwargs) -> bool:
        """Diagnostic (not part of success): target tote placed on the WRONG
        table. Distinguishes 'followed the language' from 'guessed a side'."""
        counts = self.target_tote_location_counts(*args, **kwargs)
        wrong = "right" if self._target_side == "left" else "left"
        return counts.get(f"table_{wrong}", 0) > 0

    # ------------------------------------------------------------------
    # Teleop HUD (operator needs to see the per-episode command)
    # ------------------------------------------------------------------
    def teleop_hud_lines(self, *args, **kwargs) -> list[str]:
        counts = self.target_tote_location_counts(*args, **kwargs)
        on_target = counts.get(f"table_{self._target_side}", 0) > 0
        wrong = "right" if self._target_side == "left" else "left"
        on_wrong = counts.get(f"table_{wrong}", 0) > 0
        status = "OK" if on_target else ("WRONG TABLE" if on_wrong else "...")
        return [
            f"HAND: {(self._pick_hand or '?').upper()}",
            f"TABLE: {(self._target_side or '?').upper()}",
            f"delivered: {status}",
        ]
