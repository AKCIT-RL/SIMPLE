#!/usr/bin/env python3
"""
Verify the MuJoCo -> Unity scene export against a real model.

Runs six checks, each of which has a specific way of being wrong that is very
hard to diagnose once the geometry is already in a headset:

  1. quaternion conversion       -- against MuJoCo's own mju_quat2Mat
  2. mesh face indexing          -- local to the mesh, not global
  3. triangle winding            -- mirrored basis must not invert normals
  4. export runs                 -- scene.json and OBJs are produced
  5. reassembly (the real one)   -- body pose composed with geom pose in Unity
                                    space must land where MuJoCo says it does
  6. protocol round-trip         -- encode/decode is lossless

Usage
-----
  python scripts/test_unity_export.py --mjcf path/to/scene.xml
  python scripts/test_unity_export.py --mjcf scene.xml --out-dir /tmp/unity_scene --keep

Needs only `mujoco` and `numpy`; the rest of the SIMPLE stack is not imported,
so this runs on a workstation without Isaac Sim.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import mujoco

from simple.teleop.unity import protocol
from simple.teleop.unity.coordinates import (
    positions_to_unity,
    quaternions_to_unity,
)
from simple.teleop.unity.scene_export import (
    body_names,
    export_scene,
    is_visual_geom,
    world_poses,
)

# Y/Z swap. Its own inverse, so R_unity = S @ R_mujoco @ S.
S = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])

TOL = 1e-5  # packets are float32, so this is the meaningful precision


class Failure(Exception):
    pass


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        raise Failure(name)


def mj_quat_to_mat(q) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=np.float64))
    return out.reshape(3, 3)


def unity_quat_to_mat(q) -> np.ndarray:
    """Rotation matrix for a Unity quaternion (x, y, z, w), left-handed."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_quaternion_conversion(rng) -> None:
    print("\n[1] Conversao de quaternion")
    worst = 0.0
    for _ in range(2000):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        v = rng.normal(size=3)
        lhs = unity_quat_to_mat(quaternions_to_unity(q)) @ positions_to_unity(v)
        rhs = positions_to_unity(mj_quat_to_mat(q) @ v)
        worst = max(worst, float(np.abs(lhs - rhs).max()))
    check("rotacao preservada sob troca de base", worst < 1e-6, f"erro max {worst:.2e}")


def test_mesh_indexing(model) -> None:
    print("\n[2] Indexacao de faces de mesh")
    if model.nmesh == 0:
        print("  [SKIP] modelo nao tem meshes")
        return
    overflow = []
    tightest = None  # (margin, max index, nvert) for the mesh closest to overflowing
    for i in range(model.nmesh):
        vnum = int(model.mesh_vertnum[i])
        fadr = int(model.mesh_faceadr[i])
        fnum = int(model.mesh_facenum[i])
        if fnum == 0:
            continue
        hi = int(model.mesh_face[fadr : fadr + fnum].max())
        if hi >= vnum:
            overflow.append((i, hi, vnum))
        margin = vnum - hi
        if tightest is None or margin < tightest[0]:
            tightest = (margin, hi, vnum)
    detail = (
        f"pior caso: maior indice {tightest[1]} < nvert {tightest[2]}"
        if tightest
        else "sem faces"
    )
    check("indices sao locais a cada mesh (nao globais)", not overflow, detail)


def test_winding(model) -> None:
    """A mirrored basis reverses winding; the exporter must reverse it back.

    Measured as signed volume via the divergence theorem: a closed mesh with
    outward normals encloses positive volume. If the sign flips, Unity culls
    the outside faces and the model renders inside-out.
    """
    print("\n[3] Orientacao das faces (winding)")
    from simple.teleop.unity.coordinates import mesh_to_unity

    def signed_volume(v, f):
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    checked = 0
    agree = 0
    for i in range(model.nmesh):
        vadr, vnum = int(model.mesh_vertadr[i]), int(model.mesh_vertnum[i])
        fadr, fnum = int(model.mesh_faceadr[i]), int(model.mesh_facenum[i])
        if fnum == 0:
            continue
        v = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float64)
        f = np.asarray(model.mesh_face[fadr : fadr + fnum], dtype=np.int64)
        vol_mj = signed_volume(v, f)
        if abs(vol_mj) < 1e-12:
            continue
        vu, fu = mesh_to_unity(v, f)
        vol_unity = signed_volume(np.asarray(vu, dtype=np.float64), fu)
        checked += 1
        if np.sign(vol_mj) == np.sign(vol_unity):
            agree += 1
    check(
        "volume com sinal preservado apos conversao",
        checked > 0 and agree == checked,
        f"{agree}/{checked} meshes",
    )


def test_export(model, out_dir: str) -> dict:
    print("\n[4] Execucao do export")
    manifest = export_scene(model, out_dir)
    scene_json = os.path.join(out_dir, "scene.json")
    check("scene.json escrito", os.path.isfile(scene_json))

    missing = [
        m["file"]
        for m in manifest["meshes"]
        if not os.path.isfile(os.path.join(out_dir, m["file"]))
    ]
    check(
        "todos os OBJ referenciados existem",
        not missing,
        f"{len(manifest['meshes'])} meshes",
    )

    n_visual = sum(1 for g in range(model.ngeom) if is_visual_geom(model, g))
    n_fallback = sum(1 for g in manifest["geoms"] if g.get("from_collision"))
    check(
        "geoms visuais preferidos, colisao so como fallback",
        len(manifest["geoms"]) == n_visual + n_fallback,
        f"{len(manifest['geoms'])} de {model.ngeom} geoms "
        f"({n_visual} visuais + {n_fallback} por fallback)",
    )

    # A body whose geometry is collision-only must still reach Unity -- that is
    # every manipulable object a SIMPLE scene builds.
    bodies_with_geoms = {int(model.geom_bodyid[g]) for g in range(model.ngeom)}
    bodies_exported = {g["body"] for g in manifest["geoms"]}
    check(
        "nenhum body com geometria ficou de fora",
        bodies_with_geoms == bodies_exported,
        f"{len(bodies_exported)} de {len(bodies_with_geoms)} bodies com geoms",
    )
    return manifest


def test_reassembly(model, data, manifest) -> None:
    """The end-to-end check.

    Unity will place each body at its streamed world pose, then hang static
    geoms off it at their local pose. Recomposing that here and comparing
    against MuJoCo's own resolved geom pose is the closest thing to previewing
    what the headset will show, without a headset.
    """
    print("\n[5] Remontagem no espaco Unity vs MuJoCo")
    mujoco.mj_forward(model, data)
    body_pos, body_quat = world_poses(model, data)

    by_name = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
        by_name[name] = g

    worst_pos = 0.0
    worst_rot = 0.0
    for entry in manifest["geoms"]:
        g = by_name[entry["name"]]
        b = entry["body"]

        R_body = unity_quat_to_mat(body_quat[b])
        pred_pos = body_pos[b] + R_body @ np.asarray(entry["pos"], dtype=np.float64)
        pred_rot = R_body @ unity_quat_to_mat(
            np.asarray(entry["rot"], dtype=np.float64)
        )

        true_pos = positions_to_unity(data.geom_xpos[g])
        true_rot = S @ data.geom_xmat[g].reshape(3, 3) @ S

        worst_pos = max(worst_pos, float(np.abs(pred_pos - true_pos).max()))
        worst_rot = max(worst_rot, float(np.abs(pred_rot - true_rot).max()))

    check("posicao dos geoms", worst_pos < TOL, f"erro max {worst_pos:.2e} m")
    check("orientacao dos geoms", worst_rot < TOL, f"erro max {worst_rot:.2e}")


def test_protocol(model, data, manifest) -> None:
    print("\n[6] Round-trip do protocolo")
    pos, quat = world_poses(model, data)
    scene_id = protocol.scene_id_from_names(body_names(model))
    check(
        "scene_id do manifesto bate",
        manifest["scene_id"] == scene_id,
        f"0x{scene_id:08x}",
    )

    packet = protocol.encode_state(
        frame=42, scene_id=scene_id, positions=pos, quaternions=quat
    )
    check(
        "tamanho do pacote confere",
        len(packet) == protocol.packet_size(model.nbody),
        f"{len(packet)} bytes para {model.nbody} bodies",
    )

    decoded = protocol.decode_state(packet)
    check("frame preservado", decoded["frame"] == 42)
    check("scene_id preservado", decoded["scene_id"] == scene_id)
    check(
        "poses sem perda",
        np.array_equal(decoded["positions"], pos)
        and np.array_equal(decoded["quaternions"], quat),
    )

    truncated = packet[:-4]
    try:
        protocol.decode_state(truncated)
        rejected = False
    except ValueError:
        rejected = True
    check("pacote truncado e rejeitado", rejected)

    for hz in (30, 60, 90):
        kbps = len(packet) * hz / 1000.0
        print(f"         {hz:>3} Hz -> {kbps:7.1f} KB/s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mjcf", required=True, help="MJCF scene to export")
    parser.add_argument(
        "--out-dir", default=None, help="output dir (default: a temp dir)"
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the exported scene on disk"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isfile(args.mjcf):
        print(f"MJCF nao encontrado: {args.mjcf}", file=sys.stderr)
        return 2

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="simple_unity_")
    cleanup = args.out_dir is None and not args.keep

    print(f"MJCF     : {args.mjcf}")
    print(f"Saida    : {out_dir}")

    model = mujoco.MjSpec.from_file(args.mjcf).compile()
    data = mujoco.MjData(model)
    print(
        f"Modelo   : {model.nbody} bodies, {model.ngeom} geoms, {model.njnt} joints, {model.nmesh} meshes"
    )

    rng = np.random.default_rng(args.seed)
    try:
        test_quaternion_conversion(rng)
        test_mesh_indexing(model)
        test_winding(model)
        manifest = test_export(model, out_dir)
        test_reassembly(model, data, manifest)
        test_protocol(model, data, manifest)
    except Failure as exc:
        print(f"\nFALHOU: {exc}")
        return 1
    finally:
        if cleanup:
            shutil.rmtree(out_dir, ignore_errors=True)

    print("\nTodos os testes passaram.")
    if not cleanup:
        print(f"Cena exportada em: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
