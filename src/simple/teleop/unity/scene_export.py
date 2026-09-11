"""
Export a compiled MuJoCo model as a Unity-renderable scene.

Writes into ``out_dir``:
    scene.json         -- body list, visual geoms, mesh references, scene id
    meshes/<name>.obj  -- one OBJ per visual mesh, in Unity space

Why the compiled model, not the MJCF
------------------------------------
Compilation resolves ``<default>`` class inheritance, mesh scaling and asset
paths, and -- the part that bites -- recenters every mesh's vertices on its
center of mass while baking the compensating offset into ``geom_pos`` and
``geom_quat``. On the G1, ``mesh_pos[0]`` and the pelvis ``geom_pos`` are the
same ``[0, 0, -0.076]`` for exactly this reason, and ``mesh_quat`` departs from
identity by over a radian on some links.

Reading the compiled ``MjModel`` means those adjustments are already consistent
with each other. Re-parsing the source XML would mean reproducing MuJoCo's
compiler by hand, and applying the mesh transform on top of a ``geom_pos`` that
already contains it is how a robot arrives in Unity in pieces.

Which geoms get exported
------------------------
Well-separated models carry two parallel sets of geometry. Visual geoms have
``contype == 0 and conaffinity == 0``: they are drawn but never collide.
Collision geoms are the convex hulls, spheres and capsules the solver uses, and
they look like a crude balloon animal of the robot. The G1 is built this way --
35 non-colliding meshes against 36 colliding ones.

Plenty of models are not built that way. ``MujocoSimulator._build_object`` adds
manipulable objects as convex collision meshes and nothing else, so a filter
that keeps only non-colliding geoms would drop every object in the scene and
leave the operator staring at a robot in an empty room.

So selection is per body: if a body has any non-colliding geom, those are its
visual set; if it has none, its colliding geoms are doing double duty and get
exported instead, tagged ``"from_collision": true``. Those render as the convex
decomposition the solver sees -- faceted, but present and correctly placed.
Swapping in each asset's original visual mesh is a later refinement.

``contype``/``conaffinity`` is the discriminator rather than ``group`` because
group numbering is a per-model convention, while the contact flags mean the
same thing everywhere.
"""

import json
import os

import numpy as np

from .coordinates import (
    half_extents_to_unity,
    mesh_to_unity,
    positions_to_unity,
    quaternions_to_unity,
)
from .protocol import scene_id_from_names

SCENE_FORMAT = "simple-unity-scene"
SCENE_FORMAT_VERSION = 1


def _geom_type_names():
    import mujoco

    return {
        int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
    }


def is_visual_geom(model, geom_id: int) -> bool:
    """True if ``geom_id`` is a render-only geom (takes part in no contacts)."""
    return bool(
        model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0
    )


def select_render_geoms(model, include_collision: bool = False) -> dict:
    """Choose which geoms to render, per body.

    Args:
        model: compiled ``mujoco.MjModel``.
        include_collision: export every geom, so a body's collision proxies can
            be inspected alongside its visual meshes.

    Returns:
        ``{geom_id: from_collision}``, where ``from_collision`` marks a geom
        picked only because its body offered no visual alternative.
    """
    by_body = {}
    for g in range(model.ngeom):
        by_body.setdefault(int(model.geom_bodyid[g]), []).append(g)

    selected = {}
    for geom_ids in by_body.values():
        visual = [g for g in geom_ids if is_visual_geom(model, g)]
        if include_collision:
            for g in geom_ids:
                selected[g] = not is_visual_geom(model, g)
        elif visual:
            for g in visual:
                selected[g] = False
        else:
            # No visual geometry on this body; its collision shapes are all the
            # geometry there is, so render those rather than nothing.
            for g in geom_ids:
                selected[g] = True
    return selected


def body_names(model) -> list:
    """Body names in model order. Unnamed bodies get a stable positional name."""
    import mujoco

    names = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        names.append(name if name else f"body_{i}")
    return names


def _sanitize(name: str) -> str:
    """Make ``name`` safe as a file name on Windows and POSIX alike."""
    keep = []
    for ch in name:
        keep.append(ch if (ch.isalnum() or ch in "-_.") else "_")
    out = "".join(keep).strip("._")
    return out or "unnamed"


def _write_obj(path: str, vertices, faces, name: str) -> None:
    """Write a minimal OBJ. Vertices and faces must already be in Unity space.

    Normals are omitted deliberately: Unity's ``Mesh.RecalculateNormals()``
    derives them from the winding, so shipping them would only create a second
    source of truth that can disagree with the triangles.
    """
    lines = [f"# exported from SIMPLE for Unity: {name}", f"o {_sanitize(name)}"]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    # OBJ face indices are 1-based.
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


def _geom_color(model, geom_id: int):
    """Resolve a geom's RGBA, preferring its material over its own rgba."""
    mat_id = int(model.geom_matid[geom_id])
    if mat_id >= 0:
        return [float(v) for v in model.mat_rgba[mat_id]]
    return [float(v) for v in model.geom_rgba[geom_id]]


def _geom_shape(model, geom_id: int, type_name: str, mesh_names) -> dict:
    """Describe a geom's shape in terms Unity can build directly.

    MuJoCo's ``geom_size`` means something different per type, so the ambiguity
    is resolved here rather than left for the C# side to rediscover. Capsule and
    cylinder axes need no special handling: MuJoCo orients them along local +Z,
    which the Y/Z swap turns into Unity's +Y -- already the axis Unity's own
    primitives use.
    """
    size = model.geom_size[geom_id]
    if type_name == "mesh":
        mesh_id = int(model.geom_dataid[geom_id])
        return {"type": "mesh", "mesh": mesh_names[mesh_id]}
    if type_name == "sphere":
        return {"type": "sphere", "radius": float(size[0])}
    if type_name in ("capsule", "cylinder"):
        return {
            "type": type_name,
            "radius": float(size[0]),
            "half_length": float(size[1]),
        }
    if type_name in ("box", "ellipsoid"):
        return {
            "type": type_name,
            "half_extents": half_extents_to_unity(size).tolist(),
        }
    if type_name == "plane":
        # size = (x half-extent, y half-extent, grid spacing); a zero extent
        # means the plane is infinite in that direction.
        return {
            "type": "plane",
            "half_extents": [float(size[0]), float(size[1])],
            "infinite": bool(size[0] == 0.0 or size[1] == 0.0),
        }
    return {"type": type_name, "size": [float(v) for v in size]}


def export_scene(model, out_dir: str, include_collision: bool = False) -> dict:
    """Export ``model``'s renderable geometry for Unity.

    Args:
        model: a compiled ``mujoco.MjModel``.
        out_dir: directory to write ``scene.json`` and ``meshes/`` into. Created
            if absent.
        include_collision: export every geom rather than one set per body, so a
            body's collision proxies can be inspected next to its visual meshes.
            Off by default: on the G1 it doubles the geom count and buries the
            robot in overlapping hulls.

    Returns:
        The scene manifest dict, which is also written to ``scene.json``.
    """
    import mujoco

    os.makedirs(out_dir, exist_ok=True)
    mesh_dir = os.path.join(out_dir, "meshes")

    type_names = _geom_type_names()
    names = body_names(model)

    mesh_names = []
    for i in range(model.nmesh):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
        mesh_names.append(name if name else f"mesh_{i}")

    bodies = []
    for i in range(model.nbody):
        bodies.append(
            {
                "index": i,
                "name": names[i],
                "parent": int(model.body_parentid[i]),
            }
        )

    geoms = []
    used_meshes = set()
    selected = select_render_geoms(model, include_collision=include_collision)
    for g in sorted(selected):
        from_collision = selected[g]

        type_id = int(model.geom_type[g])
        type_name = type_names.get(type_id)
        if type_name is None:
            # hfield, sdf and friends have no direct Unity primitive.
            continue

        entry = {
            "body": int(model.geom_bodyid[g]),
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            or f"geom_{g}",
            "pos": positions_to_unity(model.geom_pos[g]).tolist(),
            "rot": quaternions_to_unity(model.geom_quat[g]).tolist(),
            "rgba": _geom_color(model, g),
        }
        entry.update(_geom_shape(model, g, type_name, mesh_names))
        if from_collision:
            entry["from_collision"] = True
        if type_name == "mesh":
            used_meshes.add(int(model.geom_dataid[g]))
        geoms.append(entry)

    meshes = []
    if used_meshes:
        os.makedirs(mesh_dir, exist_ok=True)
    for mesh_id in sorted(used_meshes):
        name = mesh_names[mesh_id]
        vadr = int(model.mesh_vertadr[mesh_id])
        vnum = int(model.mesh_vertnum[mesh_id])
        fadr = int(model.mesh_faceadr[mesh_id])
        fnum = int(model.mesh_facenum[mesh_id])

        vertices = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float64)
        # mesh_face indices are local to the mesh, so no vadr offset is applied.
        faces = np.asarray(model.mesh_face[fadr : fadr + fnum], dtype=np.int64)

        vertices, faces = mesh_to_unity(vertices, faces)

        rel_path = f"meshes/{_sanitize(name)}.obj"
        _write_obj(os.path.join(out_dir, rel_path), vertices, faces, name)
        meshes.append(
            {
                "name": name,
                "file": rel_path,
                "vertices": vnum,
                "faces": fnum,
            }
        )

    manifest = {
        "format": SCENE_FORMAT,
        "version": SCENE_FORMAT_VERSION,
        "coordinate_space": "unity",
        "scene_id": scene_id_from_names(names),
        # Body poses arrive in world space, so the recommended Unity layout is a
        # flat set of GameObjects under one scene root. Nesting them by `parent`
        # and then assigning world poses per frame would work but does the
        # kinematics twice for nothing; `parent` is here for inspection and for
        # grouping in the hierarchy view.
        "pose_space": "world",
        "bodies": bodies,
        "geoms": geoms,
        "meshes": meshes,
    }

    with open(os.path.join(out_dir, "scene.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest


def world_poses(model, data):
    """Read every body's world pose, converted to Unity space.

    Args:
        model: compiled ``mujoco.MjModel``.
        data: ``mujoco.MjData`` with forward kinematics already evaluated. A
            ``mj_step`` leaves ``xpos``/``xquat`` current; after writing qpos by
            hand, call ``mj_forward`` first.
    Returns:
        ``(positions, quaternions)`` of shape ``(nbody, 3)`` and ``(nbody, 4)``.
    """
    return positions_to_unity(data.xpos), quaternions_to_unity(data.xquat)
