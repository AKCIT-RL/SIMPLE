#!/usr/bin/env python3
"""
Verify the operator-to-robot wrist alignment.

This reproduces WristsPreProcessor's arithmetic against a stand-in whose two
hand frames are mirror images, the way the G1's are. That mirroring is the
whole difficulty: it is why a single hardcoded correction makes one arm track
and the other invert, and why a test with one symmetric arm would pass while
the robot misbehaves.

Each case drives the full path -- calibrate, move the operator's hand, read
where the robot's hand went -- and asks the only question that matters: did the
robot's hand move the way the operator's did.

Usage
-----
  python scripts/test_unity_wrist_alignment.py

Needs only numpy.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from simple.teleop.unity.wrist_alignment import align_wrist_frames, solve_alignment

failures = 0

# The preprocessor's own constants, so the comparison is against what actually
# runs rather than a paraphrase of it.
RIGHT_HAND_ROTATION = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
HAND_ROTATION_CORRECTION = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)


def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    print(
        f"  [{'PASS' if condition else 'FAIL'}] {name}"
        + (f" -- {detail}" if detail else "")
    )
    if not condition:
        failures += 1


def rotation(axis: str, degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def pose(rot: np.ndarray, position) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = position
    return m


class FakeWristsPreProcessor:
    """The parts of WristsPreProcessor the alignment touches, and its maths.

    Transcribed from control/teleop/pre_processor/wrists/wrists.py: calibrate()
    stores the inverse operator pose and the robot's hand pose, and __call__
    conjugates the operator's delta into the hand frame. Anything that does not
    bear on the frame -- elbow calibration, motion scaling -- is left out.
    """

    LEFT, RIGHT = "link_LArm7", "link_RArm7"

    def __init__(self, hand_poses: dict):
        self.ee_name_list = list(hand_poses)
        self._hand_poses = hand_poses
        self.robot_world_T_init_ee = {}
        self.init_teleop_T_teleop_world = {}
        self.init_teleop_T_init_ee = {}

    def calibrate(self, wrists: dict, control_device: str) -> None:
        for ee_name in self.ee_name_list:
            side = "left" if ee_name == self.LEFT else "right"
            self.robot_world_T_init_ee[ee_name] = self._hand_poses[ee_name].copy()
            self.init_teleop_T_teleop_world[ee_name] = np.linalg.inv(wrists[side])

            self.init_teleop_T_init_ee[ee_name] = np.eye(4)
            if control_device in ["pico", "vuer"]:
                pass
            elif ee_name == self.LEFT:
                self.init_teleop_T_init_ee[ee_name][:3, :3] = HAND_ROTATION_CORRECTION
            else:
                self.init_teleop_T_init_ee[ee_name][:3, :3] = (
                    RIGHT_HAND_ROTATION @ HAND_ROTATION_CORRECTION
                )

    def __call__(self, wrists: dict) -> dict:
        out = {}
        for ee_name in self.ee_name_list:
            side = "left" if ee_name == self.LEFT else "right"
            alignment = self.init_teleop_T_init_ee[ee_name]
            delta = self.init_teleop_T_teleop_world[ee_name] @ wrists[side]
            out[ee_name] = self.robot_world_T_init_ee[ee_name] @ (
                np.linalg.inv(alignment) @ delta @ alignment
            )
        return out


def g1_like_hands() -> dict:
    """Two hand frames mirrored across the sagittal plane, as the G1's are.

    Per the Unitree URDF convention the user documented: both hands point x from
    wrist to middle finger and z from pinky to index, but y runs palm-to-back on
    the left and back-to-palm on the right. Built here by conjugating the left
    frame through the y-reflection so the pair is a genuine mirror rather than
    two arbitrary rotations.
    """
    mirror = np.diag([1.0, -1.0, 1.0])
    left_rot = rotation("z", 25.0) @ rotation("x", -70.0)
    right_rot = mirror @ left_rot @ mirror
    return {
        FakeWristsPreProcessor.LEFT: pose(left_rot, [0.2, 0.25, 0.1]),
        FakeWristsPreProcessor.RIGHT: pose(right_rot, [0.2, -0.25, 0.1]),
    }


def operator_wrists(rot: np.ndarray) -> dict:
    """Both controllers held in the same orientation.

    OpenXR reports the grip pose the same way for both hands -- the convention
    is not mirrored, the operator's hands are -- so a headset really does hand
    over one orientation for both, which is the premise the robot's mirrored
    frames collide with.
    """
    return {
        "left": pose(rot, [0.3, 0.2, 0.0]),
        "right": pose(rot, [0.3, -0.2, 0.0]),
    }


def robot_hand_motion(control_device: str, use_alignment: bool, motion) -> dict:
    """Move the operator's hands by ``motion`` and report where the robot's went."""
    wrist_rot = rotation("y", 35.0) @ rotation("x", 15.0)
    pre = FakeWristsPreProcessor(g1_like_hands())

    start = operator_wrists(wrist_rot)
    pre.calibrate(start, control_device)
    if use_alignment:
        align_wrist_frames(pre)
    before = pre(start)

    moved = {side: m.copy() for side, m in start.items()}
    for side in moved:
        moved[side][:3, 3] += motion
    after = pre(moved)

    return {ee: after[ee][:3, 3] - before[ee][:3, 3] for ee in pre.ee_name_list}


def test_solve_alignment() -> None:
    print("\n[1] Solucao do alinhamento")

    identity = solve_alignment(np.eye(3), np.eye(3))
    check("frames iguais dao identidade", np.allclose(identity, np.eye(4)))

    a = solve_alignment(rotation("z", 40.0), rotation("x", -15.0))
    check("resultado e rotacao pura", np.allclose(a[:3, 3], 0.0) and a[3, 3] == 1.0)
    check(
        "bloco rotacional e ortogonal",
        np.allclose(a[:3, :3] @ a[:3, :3].T, np.eye(3)),
    )
    check("sem reflexao (det = +1)", np.isclose(np.linalg.det(a[:3, :3]), 1.0))

    for bad, why in ((np.zeros((3, 3)), "singular"), (np.eye(2), "forma errada")):
        try:
            solve_alignment(bad, np.eye(3))
            raised = False
        except ValueError:
            raised = True
        check(f"rotacao invalida rejeitada ({why})", raised)


def test_hardcoded_corrections_break_one_arm() -> None:
    """The bug, reproduced. Neither stock branch can serve both arms."""
    print("\n[2] As correcoes fixas quebram um dos bracos")

    up = np.array([0.0, 0.0, 0.30])
    for device, label in (("vuer", "identidade"), ("unity", "correcao fixa")):
        moved = robot_hand_motion(device, use_alignment=False, motion=up)
        left = moved[FakeWristsPreProcessor.LEFT]
        right = moved[FakeWristsPreProcessor.RIGHT]
        check(
            f"{label}: pelo menos um braco erra ao levantar",
            not (
                np.allclose(left, up, atol=1e-6)
                and np.allclose(right, up, atol=1e-6)
            ),
            f"esq {np.round(left, 3)}, dir {np.round(right, 3)}",
        )


def test_alignment_fixes_both_arms() -> None:
    print("\n[3] O alinhamento calculado corrige os dois bracos")

    moves = {
        "levantar": np.array([0.0, 0.0, 0.30]),
        "ir para a direita": np.array([0.0, -0.25, 0.0]),
        "ir para frente": np.array([0.35, 0.0, 0.0]),
    }
    for label, motion in moves.items():
        moved = robot_hand_motion("vuer", use_alignment=True, motion=motion)
        for ee_name, delta in moved.items():
            side = "esquerdo" if ee_name == FakeWristsPreProcessor.LEFT else "direito"
            check(
                f"{label}: braco {side} acompanha 1:1",
                np.allclose(delta, motion, atol=1e-9),
                f"{np.round(delta, 3)} vs {np.round(motion, 3)}",
            )


def test_alignment_maps_rotation_too() -> None:
    """Translation is only half of it: the hand has to turn the right way."""
    print("\n[4] A rotacao da mao tambem acompanha")

    wrist_rot = rotation("y", 35.0) @ rotation("x", 15.0)
    turn = rotation("z", 30.0)
    pre = FakeWristsPreProcessor(g1_like_hands())

    start = operator_wrists(wrist_rot)
    pre.calibrate(start, "vuer")
    align_wrist_frames(pre)
    before = pre(start)

    turned = {side: pose(turn @ m[:3, :3], m[:3, 3]) for side, m in start.items()}
    after = pre(turned)

    for ee_name in pre.ee_name_list:
        side = "esquerdo" if ee_name == FakeWristsPreProcessor.LEFT else "direito"
        applied = after[ee_name][:3, :3] @ before[ee_name][:3, :3].T
        check(
            f"braco {side} gira igual a mao, no frame do mundo",
            np.allclose(applied, turn, atol=1e-9),
        )


def test_grip_offset_survives_the_differential() -> None:
    """A controller-local offset changes the pivot; a world-constant cannot.

    This is the distinction that matters when picking which knob to reach for.
    WAIST_FROM_HEAD is a world-frame constant: it appears on both sides of
    inv(W0) @ W and cancels exactly, which is why adding it changed nothing. A
    grip offset is a right-multiplication by a translation in the controller's
    own frame, so it conjugates the delta instead of cancelling -- and shows up
    precisely when the operator rotates the controller in place.
    """
    print("\n[6] O offset de punho sobrevive ao diferencial")

    wrist_rot = rotation("y", 35.0) @ rotation("x", 15.0)
    local_offset = np.array([0.0, 0.0, -0.06])
    world_constant = np.array([0.15, 0.0, 0.45])

    def hand_after_turning_in_place(offset=None, constant=None):
        """Rotate the controller about its own origin, without moving the hand."""
        pre = FakeWristsPreProcessor(g1_like_hands())

        def wrists(rot):
            out = operator_wrists(rot)
            for side, m in out.items():
                if offset is not None:
                    m[:3, 3] = m[:3, 3] + m[:3, :3] @ offset
                if constant is not None:
                    m[:3, 3] = m[:3, 3] + constant
            return out

        start = wrists(wrist_rot)
        pre.calibrate(start, "vuer")
        align_wrist_frames(pre)
        before = pre(start)
        after = pre(wrists(rotation("z", 45.0) @ wrist_rot))
        return {ee: after[ee][:3, 3] - before[ee][:3, 3] for ee in pre.ee_name_list}

    baseline = hand_after_turning_in_place()
    with_offset = hand_after_turning_in_place(offset=local_offset)
    with_constant = hand_after_turning_in_place(constant=world_constant)

    for ee_name in baseline:
        side = "esquerdo" if ee_name == FakeWristsPreProcessor.LEFT else "direito"
        check(
            f"constante no mundo nao muda nada ({side})",
            np.allclose(baseline[ee_name], with_constant[ee_name], atol=1e-12),
            f"{np.round(with_constant[ee_name] - baseline[ee_name], 6)}",
        )
        moved = np.linalg.norm(with_offset[ee_name] - baseline[ee_name])
        check(
            f"offset no controle move o pivo ({side})",
            moved > 0.01,
            f"{moved * 100:.1f} cm de diferenca ao girar no lugar",
        )

    # Without any offset the hand should stay put when the controller turns on
    # its own origin -- that is what makes the drift a usable calibration signal.
    for ee_name, delta in baseline.items():
        side = "esquerdo" if ee_name == FakeWristsPreProcessor.LEFT else "direito"
        check(
            f"sem offset, girar no lugar nao transladar ({side})",
            np.allclose(delta, 0.0, atol=1e-9),
            str(np.round(delta, 6)),
        )


def test_alignment_needs_calibration_first() -> None:
    print("\n[5] Sem calibracao nao ha o que alinhar")
    pre = FakeWristsPreProcessor(g1_like_hands())
    pre.ee_name_list = []
    check("preprocessador nao calibrado devolve vazio", align_wrist_frames(pre) == {})


def main() -> int:
    test_solve_alignment()
    test_hardcoded_corrections_break_one_arm()
    test_alignment_fixes_both_arms()
    test_alignment_maps_rotation_too()
    test_grip_offset_survives_the_differential()
    test_alignment_needs_calibration_first()

    print("\nTodos os testes passaram." if not failures else f"\n{failures} FALHA(S).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
