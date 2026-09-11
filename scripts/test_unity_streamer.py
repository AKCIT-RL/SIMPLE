#!/usr/bin/env python3
"""
Verify the Unity XR input path that feeds the whole-body controller.

Covers everything in ``simple.teleop.unity.streamer`` that does not need the
controller stack: parsing what Unity sends, the frame change, dead zones, yaw
integration, height clamping, edge-triggered toggles and the finger encoding.

What it deliberately does not cover is the wrist convention itself. The
arithmetic is checked -- the frame change is orthogonal, the head-relative
translation is exact -- but whether the result is the basis
``WristsPreProcessor`` expects can only be settled by moving a robot. A wrong
basis there still passes every test here.

Usage
-----
  python scripts/test_unity_streamer.py

Needs only numpy.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from simple.teleop.unity.streamer import (
    REST_LEFT_WRIST,
    UnityButtonPoller,
    UnityStreamerCore,
    UnityTrackerSource,
    apply_dead_zone,
    apply_grip_offset,
    finger_data_from_controller,
    headset_relative_wrist,
    is_usable_pose,
    matrix_from_payload,
    to_robot_frame,
    to_waist_origin,
)

failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    print(
        f"  [{'PASS' if condition else 'FAIL'}] {name}"
        + (f" -- {detail}" if detail else "")
    )
    if not condition:
        failures += 1


def pose(x=0.0, y=0.0, z=0.0) -> list:
    """A translation-only pose, flattened column-major as Unity sends it."""
    m = np.eye(4)
    m[0:3, 3] = (x, y, z)
    return m.flatten(order="F").tolist()


def tracker_message(**overrides) -> str:
    payload = {
        "head": pose(0, 1.6, 0),
        "left": pose(-0.2, 1.2, -0.3),
        "right": pose(0.2, 1.2, -0.3),
        "leftTrigger": 0.0,
        "rightTrigger": 0.0,
        "leftGrip": 0.0,
        "rightGrip": 0.0,
        "leftStickX": 0.0,
        "leftStickY": 0.0,
        "rightStickX": 0.0,
        "rightStickY": 0.0,
        "leftPrimary": False,
        "leftSecondary": False,
        "rightPrimary": False,
        "rightSecondary": False,
        "leftStickClick": False,
        "rightStickClick": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_payload_parsing() -> None:
    print("\n[1] Leitura das mensagens do canal tracker")
    source = UnityTrackerSource()

    check("mensagem valida aceita", source.feed(tracker_message()))
    check("aceita bytes tambem", source.feed(tracker_message().encode("utf-8")))
    check("JSON invalido contado, nao levantado", source.feed("{nao json") is False)
    check("tipo errado rejeitado", source.feed("[1,2,3]") is False)
    stats = source.stats()
    check("contadores batem", stats == {"received": 2, "rejected": 2}, str(stats))

    data = source.snapshot()
    check(
        "pose lida em ordem column-major",
        np.allclose(data.head_pose[0:3, 3], [0, 1.6, 0]),
        str(data.head_pose[0:3, 3]),
    )

    # Tracking blinks: a degenerate pose must not replace a good one, or the
    # arm lunges to the origin every time a controller drops out.
    before = source.snapshot().left_wrist_pose.copy()
    source.feed(tracker_message(left=[0.0] * 16))
    check(
        "pose degenerada nao substitui a ultima boa",
        np.allclose(source.snapshot().left_wrist_pose, before),
    )

    check("matriz singular rejeitada", not is_usable_pose(np.zeros((4, 4))))
    check("NaN rejeitado", not is_usable_pose(np.full((4, 4), np.nan)))
    check("identidade aceita", is_usable_pose(np.eye(4)))

    try:
        matrix_from_payload([1, 2, 3])
        raised = False
    except ValueError:
        raised = True
    check("tamanho errado de matriz e erro", raised)


def test_frame_change() -> None:
    print("\n[2] Mudanca de referencial OpenXR -> robo")
    rng = np.random.default_rng(0)

    # The change of basis must be a rotation: lengths and angles survive it, or
    # the wrist target drifts in a way no IK can absorb.
    worst = 0.0
    for _ in range(500):
        v = rng.normal(size=3)
        m = np.eye(4)
        m[0:3, 3] = v
        out = to_robot_frame(m)
        worst = max(worst, abs(np.linalg.norm(out[0:3, 3]) - np.linalg.norm(v)))
    check("preserva distancias", worst < 1e-12, f"erro max {worst:.2e}")

    rot = to_robot_frame(np.eye(4))[0:3, 0:3]
    check("parte rotacional e ortogonal", np.allclose(rot @ rot.T, np.eye(3)))
    check("sem reflexao (det = +1)", np.isclose(np.linalg.det(rot), 1.0))

    head = np.eye(4)
    head[0:3, 3] = [1.0, 2.0, 3.0]
    wrist = np.eye(4)
    wrist[0:3, 3] = [1.5, 2.0, 3.25]
    rel = headset_relative_wrist(wrist, head)
    check(
        "translacao vira relativa a cabeca",
        np.allclose(rel[0:3, 3], [0.5, 0.0, 0.25]),
        str(rel[0:3, 3]),
    )
    check("rotacao do pulso preservada", np.allclose(rel[0:3, 0:3], wrist[0:3, 0:3]))

    shifted = to_waist_origin(rel)
    check(
        "origem desce da cabeca para a cintura",
        np.allclose(shifted[0:3, 3], rel[0:3, 3] + [0.15, 0.0, 0.45]),
        str(shifted[0:3, 3]),
    )
    check(
        "rotacao sobrevive a mudanca de origem",
        np.allclose(shifted[0:3, 0:3], rel[0:3, 0:3]),
    )

    # The check that matters: an operator standing with hands at chest height
    # must produce a target the arm can actually reach. Pure algebra passes
    # whether or not the waist offset is applied -- only a reachability bound
    # notices that the target sits down by the knees.
    #
    # OpenXR is y-up and z-back, so this is a head at 1.5 m and wrists 40 cm
    # lower, 15 cm to each side, 30 cm in front.
    source = UnityTrackerSource()
    source.feed(
        tracker_message(
            head=pose(0.0, 1.5, 0.0),
            left=pose(-0.15, 1.10, -0.30),
            right=pose(0.15, 1.10, -0.30),
        )
    )
    left, right = source.wrist_poses()
    for side, target in (("esquerdo", left), ("direito", right)):
        x, y, z = target[0:3, 3]
        check(
            f"pulso {side} fica a frente do tronco",
            0.0 < x < 0.6,
            f"x = {x:.3f}",
        )
        check(
            f"pulso {side} fica na altura do tronco",
            -0.1 < z < 0.5,
            f"z = {z:.3f}",
        )
        check(
            f"pulso {side} fica ao alcance do ombro",
            np.linalg.norm([x, y, z]) < 0.75,
            f"distancia {np.linalg.norm([x, y, z]):.3f} m",
        )
    check(
        "lados nao trocam",
        left[1, 3] > 0.0 > right[1, 3],
        f"y esq {left[1, 3]:.3f}, y dir {right[1, 3]:.3f}",
    )


def test_grip_offset() -> None:
    """The controller's reported origin is not where the wrist pivots.

    Runtimes disagree about where on the controller body the pose sits --
    WebXR near the middle of the handle, Unity nearer the base. The gap is
    rigid, and the whole point is that it must rotate with the hand: a
    world-frame shift leaves the pivot wrong, so turning the wrist swings the
    target through an arc and the robot's hand orbits instead of rotating.
    """
    print("\n[3] Deslocamento da origem do controle")

    identity = np.eye(4)
    check(
        "offset zero nao altera nada",
        apply_grip_offset(identity, (0, 0, 0)) is identity,
    )

    out = apply_grip_offset(identity, (0, 0, -0.05))
    check(
        "sem rotacao, desloca no eixo pedido",
        np.allclose(out[0:3, 3], [0, 0, -0.05]),
        str(out[0:3, 3]),
    )

    # Turn the controller 180 degrees about Y: a local offset must follow it.
    turned = np.eye(4)
    turned[0:3, 0:3] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    out = apply_grip_offset(turned, (0, 0, -0.05))
    check(
        "offset acompanha a rotacao do controle",
        np.allclose(out[0:3, 3], [0, 0, 0.05]),
        str(out[0:3, 3]),
    )

    # The property that matters: with the origin on the pivot, spinning the
    # controller in place must not translate the target.
    pivot = np.array([0.0, 0.0, -0.05])
    positions = []
    for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        c, sn = np.cos(angle), np.sin(angle)
        m = np.eye(4)
        m[0:3, 0:3] = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
        # Origin orbits the true pivot, as a base-of-controller pose would.
        m[0:3, 3] = -m[0:3, 0:3] @ pivot
        positions.append(apply_grip_offset(m, pivot)[0:3, 3])
    spread = float(np.ptp(np.array(positions), axis=0).max())
    check(
        "com o offset certo, girar nao transladar",
        spread < 1e-12,
        f"dispersao {spread:.2e} m",
    )

    # And without it, the same motion drags the target around a 10 cm circle --
    # the "wrists feel wrong" symptom, measured.
    spread_raw = float(
        np.ptp(
            np.array(
                [
                    apply_grip_offset(
                        np.block(
                            [
                                [
                                    np.array(
                                        [
                                            [np.cos(a), 0, np.sin(a)],
                                            [0, 1, 0],
                                            [-np.sin(a), 0, np.cos(a)],
                                        ]
                                    ),
                                    (
                                        -np.array(
                                            [
                                                [np.cos(a), 0, np.sin(a)],
                                                [0, 1, 0],
                                                [-np.sin(a), 0, np.cos(a)],
                                            ]
                                        )
                                        @ pivot
                                    ).reshape(3, 1),
                                ],
                                [np.zeros((1, 3)), np.ones((1, 1))],
                            ]
                        ),
                        (0, 0, 0),
                    )[0:3, 3]
                    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)
                ]
            ),
            axis=0,
        ).max()
    )
    check(
        "sem offset, o alvo descreve um arco",
        spread_raw > 0.09,
        f"dispersao {spread_raw * 100:.1f} cm",
    )

    try:
        apply_grip_offset(identity, (0, 0))
        raised = False
    except ValueError:
        raised = True
    check("offset com tamanho errado e erro", raised)


def test_dead_zone() -> None:
    print("\n[3] Zona morta dos analogicos")
    check("centro vira zero", apply_dead_zone(0.05, 0.1) == 0.0)
    check("limite ainda e zero", apply_dead_zone(-0.09, 0.1) == 0.0)
    check("fim de curso vai a 1", np.isclose(apply_dead_zone(1.0, 0.1), 1.0))
    check(
        "fim de curso negativo vai a -1", np.isclose(apply_dead_zone(-1.0, 0.1), -1.0)
    )
    # Without the rescale the stick would jump straight to the dead zone value
    # the moment it leaves centre.
    check(
        "saida e continua na borda",
        abs(apply_dead_zone(0.1001, 0.1)) < 1e-3,
        f"{apply_dead_zone(0.1001, 0.1):.5f}",
    )


def test_navigation_and_height() -> None:
    print("\n[4] Navegacao e altura da base")
    source = UnityTrackerSource()
    core = UnityStreamerCore(source)

    # Unity's primary2DAxis is positive away from the operator, unlike WebXR's
    # gamepad axis. Copying VuerStreamer's negation drove the robot backwards.
    source.feed(tracker_message(leftStickY=1.0))
    out = core.poll()
    check(
        "stick para frente leva o robo para frente",
        out["control_data"]["navigate_cmd"][0] > 0,
        f"lin_vel_x={out['control_data']['navigate_cmd'][0]:.2f}",
    )

    core.reset_status()
    source.feed(tracker_message(leftStickY=-1.0))
    out = core.poll()
    check(
        "stick para tras leva para tras",
        out["control_data"]["navigate_cmd"][0] < 0,
        f"lin_vel_x={out['control_data']['navigate_cmd'][0]:.2f}",
    )

    core.reset_status()
    source.feed(tracker_message(rightStickX=-1.0))
    for _ in range(50):  # one second at the policy's 50 Hz
        out = core.poll()
    yaw = out["control_data"]["navigate_cmd"][3]
    check(
        "yaw integra para ~1 rad em 1 s",
        abs(yaw - core.MAX_ANGULAR_VEL) < 0.05,
        f"yaw={yaw:.3f}",
    )

    # Yaw is a heading: left unwrapped it grows without bound and loses
    # precision after a few minutes of turning.
    core.target_yaw = 3.10
    source.feed(tracker_message(rightStickX=-1.0))
    for _ in range(20):
        out = core.poll()
    check(
        "yaw permanece em [-pi, pi]",
        -np.pi <= out["control_data"]["navigate_cmd"][3] <= np.pi,
        f"yaw={out['control_data']['navigate_cmd'][3]:.3f}",
    )

    core.reset_status()
    source.feed(tracker_message(rightPrimary=True))
    for _ in range(500):
        out = core.poll()
    check(
        "altura satura no maximo",
        np.isclose(out["control_data"]["base_height_command"], core.HEIGHT_MAX),
        f"{out['control_data']['base_height_command']:.2f}",
    )

    source.feed(tracker_message(rightSecondary=True))
    for _ in range(500):
        out = core.poll()
    check(
        "altura satura no minimo",
        np.isclose(out["control_data"]["base_height_command"], core.HEIGHT_MIN),
        f"{out['control_data']['base_height_command']:.2f}",
    )


def test_edge_triggered_toggles() -> None:
    """Held buttons are sampled 50 times a second; only the press counts."""
    print("\n[5] Toggles por borda de subida")
    source = UnityTrackerSource()
    core = UnityStreamerCore(source)

    source.feed(tracker_message(leftSecondary=True))
    first = core.poll()["teleop_data"]["toggle_activation"]
    held = [core.poll()["teleop_data"]["toggle_activation"] for _ in range(10)]
    check("dispara uma vez ao pressionar", first is True)
    check("nao repete enquanto segurado", not any(held), f"{sum(held)} repeticoes")

    source.feed(tracker_message(leftSecondary=False))
    core.poll()
    source.feed(tracker_message(leftSecondary=True))
    check(
        "dispara de novo apos soltar",
        core.poll()["teleop_data"]["toggle_activation"] is True,
    )

    source.feed(tracker_message(rightTrigger=1.0))
    check(
        "gravacao usa o gatilho direito",
        core.poll()["data_collection_data"]["toggle_data_collection"] is True,
    )
    source.feed(tracker_message(rightGrip=1.0))
    check(
        "abortar usa o grip direito",
        core.poll()["data_collection_data"]["toggle_data_abort"] is True,
    )


def test_finger_encoding() -> None:
    print("\n[6] Codificacao dos dedos")
    THUMB, INDEX, MIDDLE, RING = 0, 5, 10, 15

    open_hand = finger_data_from_controller(False, False)
    check("formato (25, 4, 4)", open_hand.shape == (25, 4, 4), str(open_hand.shape))
    check("polegar sempre aberto", open_hand[4 + THUMB, 0, 3] == 1.0)
    check(
        "mao aberta nao fecha nenhum dedo",
        open_hand[4 + INDEX, 0, 3] == 0.0
        and open_hand[4 + MIDDLE, 0, 3] == 0.0
        and open_hand[4 + RING, 0, 3] == 0.0,
    )

    check(
        "gatilho fecha o indicador",
        finger_data_from_controller(True, False)[4 + INDEX, 0, 3] == 1.0,
    )
    check(
        "gatilho + grip fecha o medio",
        finger_data_from_controller(True, True)[4 + MIDDLE, 0, 3] == 1.0,
    )
    check(
        "so grip fecha o anelar",
        finger_data_from_controller(False, True)[4 + RING, 0, 3] == 1.0,
    )


def test_button_poller() -> None:
    """Drop and reset belong to the simulator, not to the controller.

    They are kept out of the streamer on purpose: the teleop policy has no
    business knowing an elastic band exists. Both are edge-triggered, since at
    50 Hz a level-triggered reset fires fifty times while the operator's hands
    are still closing.
    """
    print("\n[8] Botoes de nivel de simulacao")
    source = UnityTrackerSource()
    poller = UnityButtonPoller(source)

    source.feed(tracker_message(leftGrip=1.0, rightGrip=1.0))
    check("ambos os grips pedem reset", poller.poll()["reset"] is True)
    held = [poller.poll()["reset"] for _ in range(20)]
    check("segurar nao repete o reset", not any(held), f"{sum(held)} repeticoes")

    check("pedido pendente e consumido uma vez", poller.take_reset_request() is True)
    check("segunda leitura ja vem vazia", poller.take_reset_request() is False)

    source.feed(tracker_message(leftGrip=0.0, rightGrip=0.0))
    poller.poll()
    source.feed(tracker_message(leftGrip=1.0, rightGrip=1.0))
    check("soltar e apertar pede de novo", poller.poll()["reset"] is True)

    # One grip alone is the data-abort binding; it must not reset the episode.
    poller.reset_status()
    source.feed(tracker_message(leftGrip=1.0, rightGrip=0.0))
    check("um grip sozinho nao reseta", poller.poll()["reset"] is False)

    poller.reset_status()
    source.feed(tracker_message(rightStickClick=True))
    check("clique do analogico direito pede drop", poller.poll()["drop"] is True)
    check("drop e consumido uma vez", poller.take_drop_request() is True)
    check("drop nao repete", poller.take_drop_request() is False)

    # Right A raises the base. Binding drop to it as well -- as an earlier
    # version did, for want of reading the stick click Unity was already
    # sending -- would drop the robot on every height adjustment.
    poller.reset_status()
    source.feed(tracker_message(rightPrimary=True))
    check("A direito sozinho nao pede drop", poller.poll()["drop"] is False)
    check("nem deixa pedido pendente", poller.take_drop_request() is False)


def test_defaults_before_any_input() -> None:
    """Before Unity connects the controller still reads a plausible posture."""
    print("\n[7] Estado antes de qualquer mensagem")
    source = UnityTrackerSource()
    check(
        "pulso esquerdo parte da pose de descanso",
        np.allclose(source.snapshot().left_wrist_pose, REST_LEFT_WRIST),
    )
    out = UnityStreamerCore(source).poll()
    check("origem identificada como unity", out["source"] == "unity")
    check(
        "sem comando de navegacao",
        all(v == 0.0 for v in out["control_data"]["navigate_cmd"][:3]),
    )
    check(
        "contrato tem as quatro secoes",
        {"ik_data", "control_data", "teleop_data", "data_collection_data"} <= set(out),
    )
    check(
        "ik_data tem pulsos e dedos",
        {"left_wrist", "right_wrist", "left_fingers", "right_fingers"}
        == set(out["ik_data"]),
    )


def main() -> int:
    test_payload_parsing()
    test_frame_change()
    test_grip_offset()
    test_dead_zone()
    test_navigation_and_height()
    test_edge_triggered_toggles()
    test_finger_encoding()
    test_button_poller()
    test_defaults_before_any_input()

    print("\nTodos os testes passaram." if not failures else f"\n{failures} FALHA(S).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
