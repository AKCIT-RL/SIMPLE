#!/usr/bin/env python3
"""
Verify which bodies the state channel carries.

A tabletop scene is mostly furniture: tables, pads and the ground are welded to
the world and identical in every frame. Streaming them costs 28 bytes each
against a packet that has to fit in one datagram, and Tailscale's 1280-byte MTU
leaves room for about 42 bodies -- so spending slots on a table is what puts a
scene over the edge.

The rule has one subtlety worth testing rather than trusting: a body with no
joint of its own still moves when something above it does. Every link of the
robot's arm past the shoulder has zero DOFs and is carried by the joints
beneath it, so the test is over the whole parent chain, not the body.

The model is stubbed. ``is_dynamic_body`` reads two integer arrays, and
building those by hand lets the chains that matter -- a deep robot arm, a
welded table, a free-floating object -- be written down explicitly instead of
depending on a particular MJCF being installed.

Usage
-----
  python scripts/test_unity_dynamic_split.py

Needs only numpy.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from simple.teleop.unity.protocol import packet_size
from simple.teleop.unity.scene_export import dynamic_body_indices, is_dynamic_body

failures = 0

# What one SCTP payload carries over a 1280-byte path MTU. Mirrors
# bridge.MTU_SAFE_PAYLOAD; duplicated rather than imported because bridge pulls
# in the WebRTC stack.
MTU_SAFE_PAYLOAD = 1195


def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    print(
        f"  [{'PASS' if condition else 'FAIL'}] {name}"
        + (f" -- {detail}" if detail else "")
    )
    if not condition:
        failures += 1


class FakeModel:
    """The two arrays the selection reads, plus nbody.

    Bodies are declared as (parent, dofs); index 0 must be the world, as MuJoCo
    guarantees.
    """

    def __init__(self, bodies, mocap=()):
        self.nbody = len(bodies)
        self.body_parentid = np.array([p for p, _ in bodies], dtype=np.int32)
        self.body_dofnum = np.array([d for _, d in bodies], dtype=np.int32)
        # -1 everywhere except the mocap bodies, as MuJoCo fills it.
        mocapid = np.full(len(bodies), -1, dtype=np.int32)
        for slot, body in enumerate(mocap):
            mocapid[body] = slot
        self.body_mocapid = mocapid


def tabletop_scene():
    """A scene shaped like the one this runs on, named for readability."""
    names = []
    bodies = []

    def add(name, parent, dofs):
        names.append(name)
        bodies.append((parent, dofs))
        return len(bodies) - 1

    world = add("world", 0, 0)

    # Furniture: welded to the world, no joint anywhere.
    add("table1", world, 0)
    add("table2", world, 0)
    add("pad", world, 0)

    # Robot: a floating base, then a chain whose links mostly have no DOF of
    # their own. This is the case a naive body_dofnum test gets wrong.
    pelvis = add("pelvis", world, 6)
    torso = add("torso_link", pelvis, 1)
    add("d435_mount", torso, 0)  # no joint, but rides the torso
    shoulder = add("shoulder_pitch", torso, 1)
    elbow = add("elbow_pitch", shoulder, 1)
    add("wrist_roll", elbow, 1)
    add("hand_palm", elbow, 0)  # no joint, but rides the elbow

    # Free-floating objects.
    for i in range(4):
        add(f"tote_{i}", world, 6)

    return FakeModel(bodies), names


def test_chain_walk() -> None:
    print("\n[1] Regra: tem DOF na cadeia acima")

    model, names = tabletop_scene()
    index = {name: i for i, name in enumerate(names)}

    check("mundo nunca e dinamico", not is_dynamic_body(model, index["world"]))
    for furniture in ("table1", "table2", "pad"):
        check(
            f"{furniture} e estatico",
            not is_dynamic_body(model, index[furniture]),
        )

    check(
        "pelvis com base flutuante e dinamico",
        is_dynamic_body(model, index["pelvis"]),
    )
    check("junta do cotovelo e dinamica", is_dynamic_body(model, index["elbow_pitch"]))

    # The two that a body_dofnum test alone would get wrong.
    check(
        "elo sem junta proprio e dinamico se o pai se move (hand_palm)",
        is_dynamic_body(model, index["hand_palm"]),
    )
    check(
        "suporte da camera acompanha o torso (d435_mount)",
        is_dynamic_body(model, index["d435_mount"]),
    )

    check("objeto livre e dinamico", is_dynamic_body(model, index["tote_0"]))


def test_mocap_bodies_are_streamed() -> None:
    """A mocap body has no joints but is moved anyway, by direct assignment.

    The chain walk alone calls it static, and the symptom is a body frozen at
    its manifest pose while the simulator moves it -- indistinguishable from a
    stuck object until someone reads the exporter.
    """
    print("\n[2] Corpos mocap entram no stream")

    model, names = tabletop_scene()
    index = {name: i for i, name in enumerate(names)}
    target = index["pad"]

    check("sem mocap, o pad e estatico", not is_dynamic_body(model, target))

    with_mocap = FakeModel(
        list(zip(model.body_parentid.tolist(), model.body_dofnum.tolist())),
        mocap=[target],
    )
    check(
        "declarado mocap, passa a ser transmitido",
        is_dynamic_body(with_mocap, target),
    )
    check(
        "e entra na lista de pacote",
        target in dynamic_body_indices(with_mocap),
    )
    check(
        "a mobilia restante continua fora",
        index["table1"] not in dynamic_body_indices(with_mocap),
    )


def test_selection_is_ordered_and_complete() -> None:
    print("\n[3] Selecao preserva a ordem do modelo")

    model, names = tabletop_scene()
    dynamic = dynamic_body_indices(model)

    check("indices em ordem crescente", dynamic == sorted(dynamic))
    check("sem repeticoes", len(dynamic) == len(set(dynamic)))
    check("mundo fora da lista", 0 not in dynamic)

    streamed = {names[i] for i in dynamic}
    check(
        "nenhum movel ficou de fora",
        {"pelvis", "hand_palm", "d435_mount", "tote_3"} <= streamed,
        str(sorted(streamed)),
    )
    check(
        "nenhuma mobilia entrou",
        not ({"table1", "table2", "pad", "world"} & streamed),
    )

    # Every body is in exactly one of the two groups, or something is placed
    # twice or never.
    static = [i for i in range(model.nbody) if i not in set(dynamic)]
    check(
        "todo corpo esta em exatamente um grupo",
        len(static) + len(dynamic) == model.nbody,
        f"{len(static)} estaticos + {len(dynamic)} dinamicos = {model.nbody}",
    )


def test_packet_fits_the_mtu() -> None:
    """The point of the exercise, stated as a budget."""
    print("\n[4] O pacote cabe na MTU")

    model, _ = tabletop_scene()
    dynamic = dynamic_body_indices(model)

    before = packet_size(model.nbody)
    after = packet_size(len(dynamic))
    check(
        "o split reduz o pacote",
        after < before,
        f"{before} -> {after} bytes ({model.nbody} -> {len(dynamic)} corpos)",
    )
    check("cabe num datagrama", after <= MTU_SAFE_PAYLOAD, f"{after} B")

    # The real G1 scene: about 30 robot links plus a handful of objects, with
    # the furniture removed. Stated as a headroom figure so a scene that grows
    # past the limit trips this rather than the headset.
    room = (MTU_SAFE_PAYLOAD - packet_size(0)) // 28
    check("cabem pelo menos 40 corpos moveis", room >= 40, f"{room} slots")


def main() -> int:
    test_chain_walk()
    test_mocap_bodies_are_streamed()
    test_selection_is_ordered_and_complete()
    test_packet_fits_the_mtu()

    print("\nTodos os testes passaram." if not failures else f"\n{failures} FALHA(S).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
