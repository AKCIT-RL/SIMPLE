"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

------------------------------------------------------------------------------
Step 1 of the industrial-task asset pipeline (Plan A).

Parses the hand-authored Isaac Sim scene `ref_map/IH_basic.usda` and extracts the
top-level layout: for every prim under `/World` it recovers translate, rotation
(as yaw about Z, in radians), scale, the `unitsResolve` factor, and the source
asset URL (payload/reference). Prims are classified into furniture / part /
container / light / ground so downstream steps know what to turn into a
mesh-collision asset (furniture, containers) vs a graspable object (parts).

The `.usda` here is ASCII, and its top-level prims are a flat list of Xforms, so
this parser is intentionally dependency-free (no pxr / Isaac Sim boot needed) and
runs instantly. Reading the *mesh geometry* of the referenced payloads is a
separate step and does require pxr (see extract_furniture_mesh.py).

Coordinate notes:
  * metersPerUnit = 1 and upAxis = Z  -> matches MuJoCo directly.
  * `xformOp:scale:unitsResolve = (0.01, ...)` on furniture only rescales the
    *referenced mesh geometry* (authored in cm); the `translate` values are
    already in stage meters, so poses need no unit conversion.
  * USD quaternions are stored (w, x, y, z). Yaw about Z is recovered with the
    standard atan2 formula (works for the general case, not just pure-Z quats).

Usage:
    .venv/bin/python scripts/industrial/extract_ih_layout.py \
        --usda ref_map/IH_basic.usda \
        --out  data/assets/industrial/ih_layout.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any

# Classification of top-level /World prims by name prefix. Everything that is a
# piece of warehouse furniture the robot interacts with (Plan A: real mesh
# collision) goes to "furniture"; small graspable props go to "part";
# tote/bin containers go to "container".
_FURNITURE_PREFIXES = (
    "TableTrolley",
    "GravityShelfBinOrganizer",
    "MobileShelvingCart",
)
_CONTAINER_PREFIXES = ("bin_b02", "bin_b04")
_PART_PREFIXES = ("screw_95", "factory_gear", "t_connector", "longbox")
_LIGHT_PREFIXES = ("SphereLight", "DomeLight", "DistantLight", "RectLight")
_SKIP_NAMES = {"GroundPlane", "PhysicsScene"}


def _classify(name: str) -> str:
    for p in _FURNITURE_PREFIXES:
        if name.startswith(p):
            return "furniture"
    for p in _CONTAINER_PREFIXES:
        if name.startswith(p):
            return "container"
    for p in _PART_PREFIXES:
        if name.startswith(p):
            return "part"
    for p in _LIGHT_PREFIXES:
        if name.startswith(p):
            return "light"
    if name in _SKIP_NAMES:
        return "skip"
    return "other"


def _yaw_from_quat(w: float, x: float, y: float, z: float) -> float:
    """Yaw (rotation about Z) in radians from a USD quaternion (w, x, y, z)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _find_world_block(text: str) -> str:
    """Return the body of `def Xform "World" { ... }` via brace matching."""
    m = re.search(r'def\s+Xform\s+"World"\s*(?:\([^)]*\)\s*)?\{', text)
    if not m:
        raise ValueError("could not find `def Xform \"World\"` block")
    start = m.end()  # just after the opening brace
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return text[start : i - 1]


def _iter_toplevel_prims(world_body: str):
    """Yield (type_keyword, name, prim_body) for each direct child of World.

    Matches both `def Type "Name"` and `def "Name"` (typeless payload refs).
    Uses brace matching so nested `over`/`def` blocks stay inside prim_body.
    """
    prim_re = re.compile(r'\bdef\s+(?:(\w+)\s+)?"([^"]+)"\s*(?:\([^)]*\)\s*)?\{')
    i = 0
    while True:
        m = prim_re.search(world_body, i)
        if not m:
            return
        type_kw = m.group(1) or ""
        name = m.group(2)
        start = m.end()
        depth = 1
        j = start
        while j < len(world_body) and depth > 0:
            c = world_body[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        yield type_kw, name, world_body[start : j - 1]
        i = j


def _floats(s: str) -> list[float]:
    return [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]


def _parse_prim_header(name: str, full_match_region: str) -> dict[str, Any]:
    """Parse the payload/reference URL from a prim's parenthesised metadata."""
    asset_url = None
    m = re.search(r'(?:payload|references)\s*=\s*@([^@]+)@', full_match_region)
    if m:
        asset_url = m.group(1)
    return {"asset_url": asset_url}


def _parse_xform(body: str) -> dict[str, Any]:
    """Extract only this prim's *own* xformOps (before any nested block)."""
    # Cut the body at the first nested `over`/`def` so we read only the prim's
    # direct transform, not a child's. (Do NOT cut at `rel` — a `rel
    # material:binding` line can precede the xformOps, e.g. factory_gear_large.)
    cut = len(body)
    for kw in (r"\bover\b", r"\bdef\b"):
        m = re.search(kw, body)
        if m:
            cut = min(cut, m.start())
    head = body[:cut]

    out: dict[str, Any] = {
        "translate": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "units_resolve": None,
        "quat": None,
        "rotateXYZ_deg": None,
        "yaw_z_rad": 0.0,
    }

    m = re.search(r"xformOp:translate\s*=\s*\(([^)]*)\)", head)
    if m:
        out["translate"] = _floats(m.group(1))[:3]

    m = re.search(r"xformOp:scale:unitsResolve\s*=\s*\(([^)]*)\)", head)
    if m:
        out["units_resolve"] = _floats(m.group(1))[:3]

    # plain scale (skip the unitsResolve line we already consumed)
    for m in re.finditer(r"xformOp:scale\s*=\s*\(([^)]*)\)", head):
        # ensure it's not the unitsResolve occurrence
        pre = head[max(0, m.start() - 12) : m.start()]
        if "unitsResolve" in head[m.start() - 20 : m.start()] or pre.endswith(":"):
            continue
        out["scale"] = _floats(m.group(1))[:3]
        break

    m = re.search(r"xformOp:orient\s*=\s*\(([^)]*)\)", head)
    if m:
        q = _floats(m.group(1))[:4]  # (w, x, y, z)
        out["quat"] = q
        out["yaw_z_rad"] = _yaw_from_quat(*q)

    m = re.search(r"xformOp:rotateXYZ\s*=\s*\(([^)]*)\)", head)
    if m:
        r = _floats(m.group(1))[:3]
        out["rotateXYZ_deg"] = r
        out["yaw_z_rad"] = math.radians(r[2])

    return out


def parse_usda(path: str) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        text = f.read()

    world_body = _find_world_block(text)
    prims: list[dict[str, Any]] = []
    for type_kw, name, body in _iter_toplevel_prims(world_body):
        category = _classify(name)
        if category == "skip":
            continue
        # The parenthesised header sits between the name and the `{`; recover the
        # asset URL from a small window before this prim's body in world_body.
        idx = world_body.find(body)
        header_window = world_body[max(0, idx - 400) : idx]
        header = _parse_prim_header(name, header_window)
        xform = _parse_xform(body)
        prims.append(
            {
                "name": name,
                "type": type_kw or "(ref)",
                "category": category,
                **header,
                **xform,
            }
        )
    return prims


def compute_simple_frame(prims: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-express the IH_basic layout in a SIMPLE-friendly frame.

    Reference origin = the TableTrolley (the assembly bench where parts are
    picked). Everything is reported as an offset from it, preserving IH_basic's
    relative geometry. IH's axes already match SIMPLE's industrial convention
    (X = robot->bench depth, Y = along-bench lateral), so no rotation is needed;
    only a translation (and the free choice of global origin/robot spawn, made
    when the task is written). Per part *type* we also report the bounding region
    of member offsets, ready to seed target/distractor regions.
    """
    ref = next((p for p in prims if p["name"].startswith("TableTrolley")), None)
    if ref is None:
        return {}
    rx, ry = ref["translate"][0], ref["translate"][1]

    def off(p):
        return [round(p["translate"][0] - rx, 4), round(p["translate"][1] - ry, 4)]

    furniture = {
        p["name"]: {"offset_xy": off(p), "yaw_rad": round(p["yaw_z_rad"], 4)}
        for p in prims
        if p["category"] == "furniture"
    }

    # Group parts/containers by type prefix (strip trailing _NN instance suffix).
    def type_key(name: str) -> str:
        base = re.sub(r"_\d+$", "", name)
        return base

    groups: dict[str, list[list[float]]] = {}
    z_by_type: dict[str, list[float]] = {}
    for p in prims:
        if p["category"] not in ("part", "container"):
            continue
        k = type_key(p["name"])
        groups.setdefault(k, []).append(off(p))
        z_by_type.setdefault(k, []).append(round(p["translate"][2], 4))

    part_regions = {}
    for k, offs in groups.items():
        xs = [o[0] for o in offs]
        ys = [o[1] for o in offs]
        part_regions[k] = {
            "count": len(offs),
            "region_low_xy": [round(min(xs), 4), round(min(ys), 4)],
            "region_high_xy": [round(max(xs), 4), round(max(ys), 4)],
            "z_range": [min(z_by_type[k]), max(z_by_type[k])],
            "offsets": offs,
        }

    return {
        "reference": ref["name"],
        "reference_world_xy": [round(rx, 4), round(ry, 4)],
        "note": "offsets are (x=depth, y=lateral) meters from the TableTrolley center; IH axes match SIMPLE industrial convention",
        "furniture": furniture,
        "part_regions": part_regions,
    }


def summarize(prims: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[str]] = {}
    for p in prims:
        by_cat.setdefault(p["category"], []).append(p["name"])
    return {cat: names for cat, names in sorted(by_cat.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usda", default="ref_map/IH_basic.usda")
    ap.add_argument("--out", default="data/assets/industrial/ih_layout.json")
    args = ap.parse_args()

    prims = parse_usda(args.usda)
    summary = summarize(prims)
    simple_frame = compute_simple_frame(prims)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "source": args.usda,
                "prims": prims,
                "summary": summary,
                "simple_frame": simple_frame,
            },
            f,
            indent=2,
        )

    print(f"Parsed {len(prims)} prims from {args.usda}")
    for cat, names in summary.items():
        print(f"  {cat:10s} ({len(names):2d}): {', '.join(names)}")
    print(f"\nWrote layout -> {args.out}")

    # Human-readable pose table for the furniture (the Plan-A collision targets).
    print("\nFurniture poses (world frame, meters / radians):")
    for p in prims:
        if p["category"] != "furniture":
            continue
        t = p["translate"]
        print(
            f"  {p['name']:32s} pos=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f})  "
            f"yaw={math.degrees(p['yaw_z_rad']):+.1f}deg  "
            f"units_resolve={p['units_resolve']}"
        )

    # SIMPLE-frame organization (offsets from the TableTrolley = assembly bench).
    sf = simple_frame
    if sf:
        print(f"\nSIMPLE frame (offsets from {sf['reference']}, x=depth / y=lateral):")
        for fname, fv in sf["furniture"].items():
            o = fv["offset_xy"]
            print(f"  {fname:32s} dxy=({o[0]:+.3f},{o[1]:+.3f})  yaw={math.degrees(fv['yaw_rad']):+.1f}deg")
        print("  parts on bench (type -> count, region, z):")
        for k, r in sf["part_regions"].items():
            lo, hi = r["region_low_xy"], r["region_high_xy"]
            print(
                f"    {k:24s} n={r['count']}  x[{lo[0]:+.3f},{hi[0]:+.3f}] "
                f"y[{lo[1]:+.3f},{hi[1]:+.3f}]  z[{r['z_range'][0]:.3f},{r['z_range'][1]:.3f}]"
            )


if __name__ == "__main__":
    main()
