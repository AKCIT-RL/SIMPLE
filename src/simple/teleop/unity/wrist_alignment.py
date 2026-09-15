"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Solve the operator-to-robot wrist alignment instead of hardcoding it.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

The problem
-----------
``WristsPreProcessor`` is differential. At calibration it stores the operator's
wrist pose ``W0`` and the robot's hand pose ``E0``; every frame after, it maps

    E = E0 @ inv(A) @ (inv(W0) @ W) @ A

where ``A`` is a fixed rotation it calls ``init_teleop_T_init_ee`` -- the
alignment between the operator's wrist frame and the robot's hand frame. The
delta is expressed in the operator's wrist frame and applied in the robot's
hand frame, so ``A`` is what reconciles the two. Get it wrong and the arm
tracks smoothly in the wrong direction.

The preprocessor chooses ``A`` from the device *name*: identity for "pico" and
"vuer", a hardcoded pair otherwise. Both are assertions about what convention
the headset reports, which is why neither survives a new one -- Unity reports
``XRNode`` device poses, matching neither.

The solution
------------
``A`` does not have to be guessed, because what it must satisfy is known.
Writing ``R_w`` and ``R_e`` for the rotations of ``W0`` and ``E0``, a pure
translation of the operator's hand by d reaches the robot as

    R_e @ A.T @ R_w.T @ d

and the operator wants the hand to move by d. So ``R_e @ A.T @ R_w.T = I``, and

    A = R_w.T @ R_e

Rotation follows for free: substituting the same A, a hand rotation dR comes out
as ``dR @ R_e``, which turns the robot's hand by dR in the world frame.

Both R_w and R_e are already stored on the preprocessor by its own
``calibrate()``, so this needs nothing the pipeline does not already have --
only to run afterwards and overwrite the guess.

Why this is better than a per-device constant
---------------------------------------------
It holds for any headset, including ones nobody has written a branch for, and
it stops being a claim about hardware that has to be rechecked when a runtime
changes what it reports. It is also self-correcting for the G1 specifically:
the two hand frames are mirror images, so ``R_e`` differs per arm and each arm
gets the alignment it needs. A single constant cannot do that, which is exactly
how the failure showed up -- one arm tracking and the other inverted.

What it assumes
---------------
That the operator wants 1:1 world-frame motion, and that both frames agree on
which way is up. ``streamer.to_robot_frame`` establishes the second; the first
is the natural teleoperation mapping and the one the working arm already had.
"""

from __future__ import annotations

import numpy as np


def solve_alignment(
    wrist_rotation: np.ndarray, hand_rotation: np.ndarray
) -> np.ndarray:
    """The 4x4 ``init_teleop_T_init_ee`` that maps this wrist to this hand 1:1.

    Args:
        wrist_rotation: 3x3 rotation of the operator's wrist at calibration.
        hand_rotation: 3x3 rotation of the robot's hand frame at calibration.

    Returns:
        A 4x4 pure rotation. The translation block is left at zero because the
        preprocessor conjugates by this matrix -- ``inv(A) @ delta @ A`` -- and
        a translation there would leak into the delta as an orientation-
        dependent offset rather than a frame change.
    """
    for name, rotation in (("wrist", wrist_rotation), ("hand", hand_rotation)):
        if rotation.shape != (3, 3):
            raise ValueError(f"Expected a 3x3 {name} rotation, got {rotation.shape}")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError(f"The {name} rotation is not orthogonal")

    alignment = np.eye(4)
    alignment[:3, :3] = wrist_rotation.T @ hand_rotation
    return alignment


def align_wrist_frames(pre_processor) -> dict:
    """Replace a calibrated ``WristsPreProcessor``'s guessed alignments.

    Call immediately after ``calibrate()``: it reads the poses that call stored
    and overwrites the alignment it chose from the device name. Before
    calibration the dictionaries are empty and this does nothing, which is the
    honest outcome -- there is no wrist pose yet to align to.

    Args:
        pre_processor: the ``WristsPreProcessor``. Duck-typed rather than
            imported so this module stays testable without the controller
            stack, which is also what lets the test below reproduce the mirrored
            right-hand frame that caused the original bug.

    Returns:
        The alignment per end-effector name, for logging. Empty if the
        preprocessor has not been calibrated.
    """
    aligned = {}
    for ee_name in pre_processor.ee_name_list:
        # calibrate() stores the inverse, being the only form it uses.
        wrist_at_calibration = np.linalg.inv(
            pre_processor.init_teleop_T_teleop_world[ee_name]
        )
        alignment = solve_alignment(
            wrist_at_calibration[:3, :3],
            pre_processor.robot_world_T_init_ee[ee_name][:3, :3],
        )
        pre_processor.init_teleop_T_init_ee[ee_name] = alignment
        aligned[ee_name] = alignment
    return aligned
