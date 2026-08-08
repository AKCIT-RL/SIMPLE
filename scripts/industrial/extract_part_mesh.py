"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

------------------------------------------------------------------------------
Industrial-task asset pipeline: DYNAMIC graspable parts (gear, t-connector, ...).

Unlike the furniture (static, mesh collision welded to world), parts are free
rigid bodies the MuJoCo engine builds itself from `collision_meshes_mujoco`
(convex hulls -> geoms, freejoint, mass ~0.1 kg). So this emits the folder layout
the AssetManager reads, mirroring the graspnet convention:

    data/assets/industrial_parts/<name>/
        visual.obj                     # full visual mesh (meters)
        collision/convex_piece_*.obj   # VHACD convex hulls (MuJoCo collision)
        <name>.usd                     # source USD (Isaac Sim replay/render)
        stable_poses.npy               # (K,7) [x,y,z, qw,qx,qy,qz] resting poses

Scale: Isaac Props/Factory parts are authored in METERS (metersPerUnit = 1.0),
not centimeters like the warehouse furniture, so units are read from the USD by
default (--units-scale to override).

USD is read via scripts/industrial/_usd_common.py (standalone pxr, no Kit boot).

Example:
    .venv/bin/python scripts/industrial/extract_part_mesh.py \
        --url "https://.../Factory/factory_gear_large.usd" \
        --name factory_gear_large --max-hulls 16
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _usd_common import (  # noqa: E402
    decompose,
    download,
    stage_meters_per_unit,
    usd_to_trimesh,
)


def compute_stable_poses(mesh: "trimesh.Trimesh", k: int = 4) -> np.ndarray:
    """Top-k physically stable resting poses as (k,7) [x,y,z, qw,qx,qy,qz].

    Matches the convention SIMPLE's spatial DR expects (dr/spatial.py): the object
    is rotated by the quaternion and lifted so it rests on the support surface, so
    z is the *height offset* (= -min_z after applying the stable orientation, same
    as totes.py), and x/y are 0 (placement x/y come from region sampling). Only the
    stable *orientation* is taken from trimesh; z is recomputed from the mesh base.

    Falls back to a single upright "lowest point on the ground" pose if trimesh
    can't compute stable poses (e.g. non-watertight mesh).
    """
    try:
        transforms, probs = mesh.compute_stable_poses(n_samples=1)
        if len(transforms) == 0:
            raise ValueError("no stable poses")
        order = np.argsort(probs)[::-1][:k]
        poses = []
        for t in np.asarray(transforms)[order]:
            rot = np.eye(4)
            rot[:3, :3] = t[:3, :3]  # stable orientation only (drop translation)
            reoriented = mesh.copy()
            reoriented.apply_transform(rot)
            z_off = -float(reoriented.bounds[0][2])  # lift base to surface
            q = trimesh.transformations.quaternion_from_matrix(rot)  # (w, x, y, z)
            poses.append([0.0, 0.0, z_off, q[0], q[1], q[2], q[3]])
        return np.array(poses, dtype=np.float64)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[stable] fallback (identity): {e}")
        z = -float(mesh.bounds[0][2])
        return np.array([[0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="Source USD URL (S3). Omit if --usd is local.")
    ap.add_argument("--usd", help="Local USD path (overrides download).")
    ap.add_argument("--name", required=True, help="Asset name / folder.")
    ap.add_argument("--out-dir", default="data/assets/industrial_parts")
    ap.add_argument(
        "--units-scale",
        type=float,
        default=None,
        help="Override cm/m scale; default reads metersPerUnit from the USD.",
    )
    ap.add_argument("--max-hulls", type=int, default=16)
    args = ap.parse_args()

    base = os.path.join(args.out_dir, args.name)
    col_dir = os.path.join(base, "collision")
    os.makedirs(col_dir, exist_ok=True)

    usd_path = args.usd
    if usd_path is None:
        assert args.url, "provide --usd or --url"
        usd_path = download(args.url, os.path.join(base, f"{args.name}.usd"))
    elif not os.path.exists(os.path.join(base, f"{args.name}.usd")):
        shutil.copy(usd_path, os.path.join(base, f"{args.name}.usd"))

    units = args.units_scale if args.units_scale is not None else stage_meters_per_unit(usd_path)
    print(f"[mesh] tessellating {usd_path} (units_scale={units})")
    mesh = usd_to_trimesh(usd_path, units)
    ext = mesh.bounds[1] - mesh.bounds[0]
    print(
        f"[mesh] {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
        f"size(m)=({ext[0]:.3f},{ext[1]:.3f},{ext[2]:.3f}), watertight={mesh.is_watertight}"
    )

    vis_obj = os.path.join(base, "visual.obj")
    mesh.export(vis_obj)
    print(f"[vis] -> {vis_obj}")

    print(f"[vhacd] decomposing (max_hulls={args.max_hulls}) ...")
    hulls = decompose(mesh, args.max_hulls)
    print(f"[vhacd] {len(hulls)} convex hulls")
    for i, h in enumerate(hulls):
        h.export(os.path.join(col_dir, f"convex_piece_{i}.obj"))
    print(f"[col] -> {col_dir}/convex_piece_*.obj")

    poses = compute_stable_poses(mesh, k=4)
    np.save(os.path.join(base, "stable_poses.npy"), poses)
    print(f"[stable] {len(poses)} stable pose(s) -> stable_poses.npy  (top z={poses[0,2]:.4f})")
    print("[done]")


if __name__ == "__main__":
    main()
