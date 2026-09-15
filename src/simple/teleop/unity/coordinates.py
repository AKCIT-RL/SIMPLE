"""
Coordinate conversion between MuJoCo and Unity.

MuJoCo is right-handed with Z up; Unity is left-handed with Y up. The change of
basis is a Y/Z swap, which also flips handedness:

    position:    (x, y, z)_mj    -> (x, z, y)_unity
    quaternion:  (w, x, y, z)_mj -> (x, z, y, -w)_unity

Note the quaternion is reordered as well: MuJoCo stores scalar-first (w, x, y, z)
while Unity stores scalar-last (x, y, z, w). Negating w is what accounts for the
handedness flip -- the same rotation, expressed in a mirrored basis, turns the
other way.

Everything that crosses the wire to Unity is converted here, so the C# side can
assign values straight into ``Transform`` without applying a conversion of its
own. Two implementations of this mapping, one per language, is precisely how the
robot ends up subtly inside-out three weeks from now.

The quaternion mapping is verified against MuJoCo's own ``mju_quat2Mat`` in
``tests/test_coordinates.py``.
"""

import numpy as np


def positions_to_unity(pos):
    """Convert MuJoCo positions to Unity space.

    Args:
        pos: ``(..., 3)`` array of MuJoCo positions.
    Returns:
        ``(..., 3)`` float32 array of Unity positions.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape[-1] != 3:
        raise ValueError(f"Expected trailing dimension 3, got {pos.shape}")
    out = np.empty_like(pos)
    out[..., 0] = pos[..., 0]
    out[..., 1] = pos[..., 2]
    out[..., 2] = pos[..., 1]
    return out.astype(np.float32)


def quaternions_to_unity(quat):
    """Convert MuJoCo quaternions to Unity space.

    Args:
        quat: ``(..., 4)`` array of MuJoCo quaternions, scalar-first (w, x, y, z).
    Returns:
        ``(..., 4)`` float32 array of Unity quaternions, scalar-last (x, y, z, w).
    """
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected trailing dimension 4, got {quat.shape}")
    out = np.empty_like(quat)
    out[..., 0] = quat[..., 1]  # x
    out[..., 1] = quat[..., 3]  # z -> y
    out[..., 2] = quat[..., 2]  # y -> z
    out[..., 3] = -quat[..., 0]  # -w
    return out.astype(np.float32)


def half_extents_to_unity(size):
    """Convert MuJoCo box/ellipsoid half-extents to Unity space.

    Half-extents are axis-aligned lengths, so they take the same Y/Z swap as a
    position but never the sign flip.
    """
    size = np.asarray(size, dtype=np.float64)
    if size.shape[-1] != 3:
        raise ValueError(f"Expected trailing dimension 3, got {size.shape}")
    out = np.empty_like(size)
    out[..., 0] = size[..., 0]
    out[..., 1] = size[..., 2]
    out[..., 2] = size[..., 1]
    return out.astype(np.float32)


def mesh_to_unity(vertices, faces):
    """Convert a MuJoCo mesh to Unity space.

    Swapping two axes mirrors the mesh, which reverses triangle winding. Left
    uncorrected, every normal points inward, Unity's backface culling discards
    the outer surface and the model renders inside-out. Reversing the index
    order of each triangle restores the original winding.

    Args:
        vertices: ``(nvert, 3)`` array of MuJoCo vertex positions.
        faces: ``(nface, 3)`` integer array of triangle vertex indices.
    Returns:
        ``(vertices, faces)`` in Unity space.
    """
    vertices = positions_to_unity(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"Expected (nface, 3) faces, got {faces.shape}")
    return vertices, faces[:, ::-1].copy()
