"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

G1SonicFixedHand -- Unitree G1 with the rigid *fixed* hand (the rubber hand,
no finger articulation). Behaves exactly like ``G1Sonic`` (body WBC + the
``decoupled_wbc`` teleop path) EXCEPT the hand is ignored everywhere:

* ``NUM_HAND_JOINTS = 0`` (forced in ``__init__`` on a copy of the WBC config),
  so gear_sonic and ``g1_sonic.apply_action`` skip every hand branch -- those
  are already guarded with ``if self.num_hand_dof > 0``, so NO control logic
  changes, only config/overrides live here.
* the finger (eef) controllers are no-ops (empty ``joint_names``);
* the arm end-effector is the wrist link (the fixed-hand MJCF has no
  ``*_hand_palm_link``).

Deploy note: the real robot has no finger actuators, so the sim2real body
control (29 DOF) maps 1:1 -- there is simply nothing hand-side to map.
"""

from typing import Any, List

from simple.core.controller import ControllerCfg
from simple.robots.controllers.combo import WholeBodyEEFControllerCfg
from simple.robots.controllers.eef import DexHandEEFControllerCfg
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.registry import RobotRegistry
from simple.robots.g1_sonic import (
    G1Sonic,
    LEFT_LEG_JOINTS,
    RIGHT_LEFT_JOINTS,   # NB: original spelling in g1_sonic (right-leg joints)
    WAIST_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
)

# Body-only joint set (legs + waist + arms). The fixed hand has 0 actuated DOF,
# so -- unlike G1Sonic's WHOLE_BODY_JOINTS -- there are no finger joints here.
BODY_JOINTS = (
    LEFT_LEG_JOINTS + RIGHT_LEFT_JOINTS + WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
)


@RobotRegistry.register("g1_sonic_fixedhand")
class G1SonicFixedHand(G1Sonic):
    """G1 with the rigid fixed (rubber) hand -- body-only control, hand ignored."""

    uid: str = "g1_sonic_fixedhand"
    label: str = "Unitree G1 Wholebody (fixed hand)"

    dof: int = 29
    wholebody_dof: int = 29        # body only (no hand DOF)
    hand_dof: int = 0

    # Rubber/fixed-hand model: 29 body joints, 0 finger joints (ends at wrist).
    mjcf_path: str = "robots/g1/g1_29dof.xml"
    # Isaac render USD: download the fixed-hand USD from Hugging Face
    # (unitreerobotics/unitree_model : G1/29dof/usd/g1_29dof_rev_1_0/) into
    # data/robots/g1/g1_29dof_fixedhand/ . Not needed for the MuJoCo path.
    usd_path: str = "robots/g1/g1_29dof_fixedhand/g1_29dof_rev_1_0.usd"
    # USD root-prim namespace (Isaac only) -- verify against the downloaded USD.
    robot_ns: str = "g1_29dof"

    # --- curobo: NOT USED (teleop only) --------------------------------------
    # We do not use curobo motion planning here. But G1Sonic inherits
    # CuRoboMixin, whose __init__ *loads* robot_cfg eagerly (the IK solver stays
    # lazy and is never built on this robot). So robot_cfg must point at a
    # loadable yaml even though it is never exercised. We reuse the existing dex3
    # curobo config as an inert, loadable placeholder (option A). If a body-only
    # curobo config or a lazy robot_cfg load is added later, swap this out.
    robot_cfg: Any = "robots/g1/curobo/g1_29dof_with_dex3.yml"

    joint_names = BODY_JOINTS
    hand_names: List[str] = []
    init_joint_states: dict = dict(zip(BODY_JOINTS, [0.0] * len(BODY_JOINTS)))

    # Arm EE / hand / wrist-cam links. The fixed-hand MJCF ends at
    # ``*_wrist_yaw_link`` (no ``*_hand_palm_link``), so point IK/EE there.
    # wrist_cam_link is unused (tasks use the head stereo camera) -- kept only
    # so the attribute resolves; a wrist camera is never configured.
    LEFT_ARM_EE_LINK: str = "left_wrist_yaw_link"
    RIGHT_ARM_EE_LINK: str = "right_wrist_yaw_link"
    eef_prim_path: str = "right_wrist_yaw_link"
    hand_prim_path: str = "right_wrist_yaw_link"
    wrist_cam_link: str = "right_wrist_yaw_link"

    # Body controllers identical to G1Sonic; the finger (eef) controllers are
    # no-ops: empty joint_names -> dof 0, nothing to bind, empty action space.
    controller_cfg: ControllerCfg = WholeBodyEEFControllerCfg(
        left_leg=PDJointPosControllerCfg(
            joint_names=LEFT_LEG_JOINTS, init_qpos=[0.0] * len(LEFT_LEG_JOINTS)
        ),
        right_leg=PDJointPosControllerCfg(
            joint_names=RIGHT_LEFT_JOINTS, init_qpos=[0.0] * len(RIGHT_LEFT_JOINTS)
        ),
        waist=PDJointPosControllerCfg(
            joint_names=WAIST_JOINTS, init_qpos=[0.0] * len(WAIST_JOINTS)
        ),
        left_arm=PDJointPosControllerCfg(
            joint_names=LEFT_ARM_JOINTS, init_qpos=[0.0] * len(LEFT_ARM_JOINTS)
        ),
        right_arm=PDJointPosControllerCfg(
            joint_names=RIGHT_ARM_JOINTS, init_qpos=[0.0] * len(RIGHT_ARM_JOINTS)
        ),
        left_eef=DexHandEEFControllerCfg(joint_names=[], init_qpos=[], close_qpos=[]),
        right_eef=DexHandEEFControllerCfg(joint_names=[], init_qpos=[], close_qpos=[]),
    )

    def __init__(self, sonic_config: dict, **kwargs) -> None:
        # Force a hand-less WBC config so gear_sonic reports 0 hand DOF and the
        # torque arrays are body-sized (NUM_JOINTS body is left untouched).
        #
        # motor_effort_limit_list needs a careful remap, NOT a [:29] slice. That
        # list (length 43) is laid out in the DEX3 kinematic-TREE order:
        #   [legs+waist: 15][left arm: 7][left hand: 7][right arm: 7][right hand: 7]
        # The fixed hand has no fingers, so its 29 body torques are indexed in
        # BODY order [legs+waist][left arm][right arm]. Slicing [:29] would take
        # [legs+waist][left arm][LEFT HAND] and feed the left-hand limits
        # (2.45 / 0.7 N.m) to the RIGHT ARM -> the right arm can't hold against
        # gravity and the hand droops. Rebuild the 29 body limits by dropping the
        # two 7-joint hand blocks. (MOTOR_KP/MOTOR_KD are already length-29 in
        # BODY order, so only the effort list is affected.)
        full = list(sonic_config["motor_effort_limit_list"])
        n_hand = sonic_config.get("NUM_HAND_JOINTS", 7)  # per hand (7), pre-override
        n_lw, n_arm = 15, 7  # G1: 12 legs + 3 waist; 7 joints per arm
        body_effort = (
            full[0 : n_lw + n_arm]                                        # legs+waist+left arm
            + full[n_lw + n_arm + n_hand : n_lw + 2 * n_arm + n_hand]      # right arm (skip left hand)
        )
        n_body = sonic_config["NUM_JOINTS"]
        assert len(body_effort) == n_body, (len(body_effort), n_body)
        sonic_config = {
            **sonic_config,
            "NUM_HAND_JOINTS": 0,
            "NUM_HAND_MOTORS": 0,
            "motor_effort_limit_list": body_effort,
        }
        super().__init__(sonic_config, **kwargs)
