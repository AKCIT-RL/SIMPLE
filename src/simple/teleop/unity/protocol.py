"""
Binary wire format for per-frame scene state, Python -> Unity.

One packet carries the world pose of every body in the scene, already converted
to Unity's coordinate space by ``coordinates``.

Header (20 bytes, little-endian):
    [4s magic   ]  b"SUST"
    [H  version ]  protocol version (currently 1)
    [H  flags   ]  reserved, must be 0
    [I  scene_id]  FNV-1a hash of the scene's body-name list
    [I  frame   ]  monotonic frame counter, from ``MujocoSimulator.render_step``
    [H  nbody   ]  number of body poses that follow
    [H  _pad    ]  reserved, must be 0

Body block (nbody * 28 bytes):
    [7f]  px py pz qx qy qz qw

Bodies are positional: index ``i`` in the packet is index ``i`` in the
``bodies`` list of the ``scene.json`` written by ``scene_export``. ``scene_id``
guards that pairing -- if Unity has a different scene loaded than the one being
streamed, the hash mismatches and the client can refuse the packet instead of
rendering a G1 whose forearm is being driven by a coffee mug.

Why world poses instead of joint angles
---------------------------------------
MuJoCo has already run forward kinematics by the time state is read, so sending
resolved body poses means Unity needs no kinematic model of its own: no joint
axes to map, no rest rotations to calibrate, no per-joint sign conventions to
discover by trial. Robot links, free-floating objects and articulated parts all
arrive through one uniform path, and the renderer cannot drift out of sync with
the physics.

The cost is bandwidth: 28 bytes per body rather than 4 bytes per joint. For a
31-body G1 that is 884 bytes per frame, about 53 KB/s at 60 Hz -- roughly an
order of magnitude below the ~500 KB/s the stereo video path consumes today.
"""

import struct

import numpy as np

MAGIC = b"SUST"
VERSION = 1
HEADER_FORMAT = "<4sHHIIHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
BODY_SIZE = 7 * 4

assert HEADER_SIZE == 20, HEADER_SIZE


def scene_id_from_names(body_names) -> int:
    """Hash a body-name list into the 32-bit scene identity carried by packets.

    FNV-1a over the newline-joined names. Chosen over ``hash()`` because Python
    salts string hashing per process, so ``hash()`` would produce a different
    id on every run and never match what Unity was told at export time.
    """
    data = "\n".join(body_names).encode("utf-8")
    h = 0x811C9DC5
    for byte in data:
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h


def encode_state(frame: int, scene_id: int, positions, quaternions) -> bytes:
    """Serialise one frame of body poses.

    Args:
        frame: monotonic frame counter.
        scene_id: value from ``scene_id_from_names`` for the streamed scene.
        positions: ``(nbody, 3)`` Unity-space positions.
        quaternions: ``(nbody, 4)`` Unity-space quaternions, scalar-last.
    Returns:
        The encoded packet.
    """
    positions = np.asarray(positions, dtype=np.float32)
    quaternions = np.asarray(quaternions, dtype=np.float32)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected (nbody, 3) positions, got {positions.shape}")
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Expected (nbody, 4) quaternions, got {quaternions.shape}")
    if len(positions) != len(quaternions):
        raise ValueError(
            f"positions/quaternions length mismatch: {len(positions)} vs {len(quaternions)}"
        )
    if len(positions) > 0xFFFF:
        raise ValueError(f"Too many bodies for a uint16 count: {len(positions)}")

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        0,
        scene_id & 0xFFFFFFFF,
        frame & 0xFFFFFFFF,
        len(positions),
        0,
    )
    # Interleave into (nbody, 7) so each body is one contiguous 28-byte run.
    block = np.empty((len(positions), 7), dtype=np.float32)
    block[:, 0:3] = positions
    block[:, 3:7] = quaternions
    return header + block.tobytes()


def decode_state(buf: bytes) -> dict:
    """Deserialise a packet produced by ``encode_state``.

    Returns:
        dict with keys ``version``, ``flags``, ``scene_id``, ``frame``,
        ``positions`` ``(nbody, 3)`` and ``quaternions`` ``(nbody, 4)``.
    Raises:
        ValueError on malformed input.
    """
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"Packet too small: {len(buf)} bytes")

    magic, version, flags, scene_id, frame, nbody, _pad = struct.unpack_from(
        HEADER_FORMAT, buf, 0
    )
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"Unsupported protocol version: {version}")

    expected = HEADER_SIZE + nbody * BODY_SIZE
    if len(buf) != expected:
        raise ValueError(
            f"Length mismatch: header declares {nbody} bodies "
            f"({expected} bytes), got {len(buf)}"
        )

    block = np.frombuffer(buf, dtype=np.float32, count=nbody * 7, offset=HEADER_SIZE)
    block = block.reshape((nbody, 7))
    return {
        "version": version,
        "flags": flags,
        "scene_id": scene_id,
        "frame": frame,
        "positions": block[:, 0:3].copy(),
        "quaternions": block[:, 3:7].copy(),
    }


def packet_size(nbody: int) -> int:
    """Bytes on the wire for a scene of ``nbody`` bodies."""
    return HEADER_SIZE + nbody * BODY_SIZE
