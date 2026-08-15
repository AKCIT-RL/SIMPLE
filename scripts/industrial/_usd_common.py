"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

------------------------------------------------------------------------------
Shared USD -> mesh helpers for the industrial asset pipeline.

pxr (USD) is read standalone from the libraries Isaac Sim already ships
(`omni.usd.libs`); this does NOT boot the full SimulationApp / Kit.
`_bootstrap_pxr()` puts those libs on the path and re-execs the *entry* script
once if pxr isn't importable yet, so importing this module Just Works from any of
the extract_*.py scripts.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys


def _bootstrap_pxr() -> None:
    """Put Isaac Sim's bundled USD runtime on the path (no Kit boot), re-exec once."""
    if importlib.util.find_spec("pxr") is not None:
        return
    spec = importlib.util.find_spec("isaacsim")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "pxr not importable and isaacsim package not found; cannot locate USD libs."
        )
    isaac_dir = list(spec.submodule_search_locations)[0]
    libs = sorted(glob.glob(os.path.join(isaac_dir, "extscache", "omni.usd.libs-*")))
    libs = [d for d in libs if os.path.isdir(os.path.join(d, "pxr"))]
    if not libs:
        raise RuntimeError("could not find omni.usd.libs under isaacsim/extscache")
    libdir = libs[0]
    if os.environ.get("_IH_PXR_BOOTSTRAPPED") == "1":
        raise RuntimeError(f"pxr still not importable after adding {libdir}")
    env = dict(os.environ)
    env["_IH_PXR_BOOTSTRAPPED"] = "1"
    env["PYTHONPATH"] = libdir + os.pathsep + env.get("PYTHONPATH", "")
    ld = f"{libdir}/bin:{libdir}/pxr"
    env["LD_LIBRARY_PATH"] = ld + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_bootstrap_pxr()

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402  # type: ignore  # runtime-provided by _bootstrap_pxr()


def download(url: str, dest: str) -> str:
    if os.path.exists(dest):
        print(f"[dl] cached: {dest}")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    import urllib.request

    print(f"[dl] {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[dl] -> {dest} ({os.path.getsize(dest)} bytes)")
    return dest


def stage_meters_per_unit(usd_path: str) -> float:
    """The asset's authored metersPerUnit (1.0 = meters, 0.01 = centimeters)."""
    stage = Usd.Stage.Open(usd_path)
    return float(UsdGeom.GetStageMetersPerUnit(stage))


def _triangulate(counts, indices):
    """Fan-triangulate polygons given USD faceVertexCounts / faceVertexIndices."""
    tris = []
    k = 0
    for c in counts:
        face = indices[k : k + c]
        for j in range(1, c - 1):
            tris.append((face[0], face[j], face[j + 1]))
        k += c
    return np.asarray(tris, dtype=np.int64)


def usd_to_trimesh(usd_path: str, units_scale: float) -> "trimesh.Trimesh":
    """Merge every Mesh prim (baked to world space, scaled to meters)."""
    stage = Usd.Stage.Open(usd_path)
    assert stage, f"failed to open {usd_path}"
    tc = Usd.TimeCode.Default()
    meshes = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get(tc)
        counts = mesh.GetFaceVertexCountsAttr().Get(tc)
        idx = mesh.GetFaceVertexIndicesAttr().Get(tc)
        if not pts or not counts or not idx:
            continue
        verts = np.array(pts, dtype=np.float64)
        # local -> world (USD is row-vector: world = pt * M)
        m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(tc)).T
        h = np.c_[verts, np.ones(len(verts))]
        verts = (h @ m.T)[:, :3] * units_scale
        faces = _triangulate(np.array(counts), np.array(idx))
        meshes.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
    assert meshes, f"no meshes found in {usd_path}"
    merged = trimesh.util.concatenate(meshes)
    merged.merge_vertices()
    return merged


def decompose(mesh: "trimesh.Trimesh", max_hulls: int):
    """VHACD convex decomposition; returns a list of convex Trimesh hulls."""
    try:
        parts = mesh.convex_decomposition(maxConvexHulls=max_hulls)
    except TypeError:
        parts = mesh.convex_decomposition()
    if isinstance(parts, trimesh.Trimesh):
        parts = [parts]
    out = []
    for p in parts:
        if isinstance(p, trimesh.Trimesh):
            out.append(p)
        else:
            out.append(trimesh.Trimesh(vertices=p["vertices"], faces=p["faces"]))
    return out
