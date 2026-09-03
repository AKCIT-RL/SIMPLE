"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Render a robot's CuRobo self-collision spheres (spheres.yml) to PNG from several camera angles,
offscreen (no display needed), one distinct color per link. Supersedes
`scripts/render_widowx_collision_spheres.py` (WidowX-only, single sphere color) -- covers Franka
FR3 too, and per-link coloring makes it possible to tell apart, at a glance, "spheres from the
same link touching in a chain" (expected -- see `generate_spheres.py`'s MIN_OVERLAP, which
deliberately overlaps consecutive spheres of one link so the chain has no gaps) from "spheres
from two different links touching" (only the latter is checked by curobo's self-collision cost --
`SelfCollisionCost` skips same-link pairs entirely -- and only the latter is a potential problem).

WHY (background)
-----------------
`docs/teleop_simple_study/widowx_ai_integration_status.md` (section 12+) investigates a
self-collision false-positive bug on WidowX AI: curobo's `spheres.yml` (originally auto-fit
against the robot's detailed STL meshes) over-approximated the true collision volume for several
links. `data/robots/widowx_ai/generate_spheres.py` now fits most links analytically against
wxai_follower.xml's own hand-authored collision primitives instead, which fixed most of it but
left a handful of residual cross-link violations (base_link/link_2, link_4/link_6,
link_6/gripper_left, link_1/link_4 -- see that doc section for the exact numbers). Franka's own
spheres.yml (curobo's bundled NVIDIA config) is included here purely as a visual reference for
"what a reasonably tight, non-broken fit looks like" -- Franka was never found to have this bug.

HOW SPHERES ARE PLACED
-----------------------
`spheres.yml` centers are in each link's own LOCAL body frame -- the same frame
`MjSpec.body(name).add_geom(pos=...)` uses for a geom parented to that body, so no FK/transform
math is needed: each sphere becomes a literal child geom of the matching MuJoCo body. Two name
mismatches needed explicit mapping (BODY_NAME_MAP below), found by direct inspection:
- Franka: spheres.yml uses `fr3_link0`..`fr3_link7`, `fr3_hand`, `fr3_leftfinger`/
  `fr3_rightfinger`; panda.xml bodies are `link0`..`link7`, `hand`, `left_finger`/`right_finger`
  (no `fr3_` prefix, underscore in the finger names). `attached_object` (a placeholder curobo
  uses for objects grasped mid-episode, not a real robot link) is skipped.
- WidowX AI: `gripper_left`/`gripper_right` are separate links in the URDF, but wxai_follower.xml
  draws their mesh directly inside `carriage_left`/`carriage_right` (no separate body). The
  URDF's fixed joint between them has zero offset, so their spheres parent straight onto
  carriage_left/right with no coordinate correction.

Per-link colors come from an HSV sweep at the golden angle (`_color_for_index`), seeded by each
robot's own sorted link name list -- deterministic (same link always gets the same color across
runs) and visually well-separated regardless of how many links a robot has.

Two images per view: `_mesh.png` (opaque real mesh alone, size/shape reference) and
`_spheres.png` (spheres alone, opaque, color-coded by link). Rendering both as one translucent
overlay was tried first and discarded -- blending translucent spheres under a translucent mesh
under bright lighting desaturates everything toward washed-out gray, harder to read than either
alone.

USAGE
-----
    MUJOCO_GL=egl .venv/bin/python scripts/render_collision_spheres.py

Writes to --out-dir (default: .render-output/collision_spheres/), for each of Franka and WidowX
AI (WidowX rendered at both `rest` and `close-high`): 2 PNGs (mesh, spheres) x 4 views
(front, side, top, iso).
    --robot franka|widowx_ai   Render only one robot instead of both.
"""

from __future__ import annotations

import colorsys
import os
from pathlib import Path
from typing import Annotated

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import typer
import yaml
from PIL import Image

MESH_GROUP = 2  # matches both MJCFs' own "visual" default class
# NOT 3 -- both panda.xml and wxai_follower.xml already use group 3 for their own real
# (invisible-by-default) "collision" class primitives; sharing it would make our visualization
# spheres untoggleable independently of those (confirmed by grepping both MJCFs).
SPHERE_GROUP = 4

VIEWS = {
    # (azimuth_deg, elevation_deg, distance)
    "front": (90, -15, 1.1),
    "side": (0, -15, 1.1),
    "top": (90, -89, 1.1),
    "iso": (45, -25, 1.1),
}

ROBOTS = {
    "franka": dict(
        mjcf="data/robots/franka_fr3/panda.xml",
        spheres="data/robots/franka_fr3/spheres.yml",
        body_map={
            **{f"fr3_link{i}": f"link{i}" for i in range(8)},
            "fr3_hand": "hand",
            "fr3_leftfinger": "left_finger",
            "fr3_rightfinger": "right_finger",
        },
        skip_links={"attached_object"},
        qpos_presets={
            "default": dict(
                zip(
                    [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"],
                    [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0, 0.04, 0.04],
                )
            ),
        },
        lookat={"default": [0.1, 0.0, 0.4]},
    ),
    "widowx_ai": dict(
        mjcf="data/robots/widowx_ai/wxai_follower.xml",
        spheres="data/robots/widowx_ai/spheres.yml",
        body_map={"gripper_left": "carriage_left", "gripper_right": "carriage_right"},
        skip_links=set(),
        qpos_presets={
            # right_carriage_joint is a MuJoCo <equality> mimic of left_carriage_joint
            # (polycoef="0 1 0 0 0", i.e. exactly equal -- confirmed by reading wxai_follower.xml
            # directly) -- mj_forward alone does NOT resolve <equality> constraints (that needs
            # mj_step's constraint solver, which never runs in this render-only script), so it
            # must be set explicitly here too. Leaving it out silently left the right gripper
            # finger's carriage at its qpos=0 (closed) default while the left one opened to
            # 0.044, making gripper_right visibly off-center relative to gripper_left in earlier
            # renders -- a rendering bug, not a spheres.yml asymmetry (spheres.yml's gripper_left/
            # gripper_right entries were already correctly mirrored; verify with `diff` if in doubt).
            "rest": dict(
                zip([f"joint_{i}" for i in range(6)] + ["left_carriage_joint", "right_carriage_joint"], [0.0] * 6 + [0.044, 0.044])
            ),
            # scripts/test_widowx_ai_curobo_ik.py's "close-high" waypoint, solved qpos (see that
            # script's WAYPOINTS docstring for provenance).
            "close-high": dict(
                zip(
                    [f"joint_{i}" for i in range(6)] + ["left_carriage_joint", "right_carriage_joint"],
                    [0.0, 0.13631485402584076, 0.8843041062355042, -0.7479904890060425, 0.0, 0.0, 0.044, 0.044],
                )
            ),
        },
        lookat={"rest": [0.0, 0.0, 0.12], "close-high": [0.0, 0.0, 0.2]},
    ),
    "viperx": dict(
        mjcf="data/robots/viperx/viperx.xml",
        spheres="data/robots/viperx/curobo/spheres/viperx.yml",
        # Unlike Franka/WidowX AI's zero-offset body_map entries, these three curobo collision
        # links have a genuinely nonzero fixed-joint offset from their MJCF parent body (confirmed
        # directly in viperx.urdf) -- left_gripper_camera and both left_custom_finger_*_link
        # links are separate URDF links purely for curobo's own kinematics, with no matching body
        # in viperx.xml at all (the MJCF draws their meshes directly inside left_gripper_base /
        # left_*_finger_link). All three joints are pure single-axis rotations about local X (no
        # ambiguity in interpreting the URDF's rpy convention), so each tuple below is
        # (parent_body_name, translation, rotation-about-X in radians); build_spec applies
        # center' = Rx(theta) @ center + translation before parenting the sphere geom.
        body_map={
            "left_gripper_camera": ("left_gripper_base", [0.0, -0.0824748, -0.0095955], -0.4363331),
            "left_custom_finger_left_link": ("left_left_finger_link", [0.0141637, 0.0211727, 0.06], 1.5707963),
            "left_custom_finger_right_link": ("left_right_finger_link", [0.0141637, -0.0211727, 0.0597067], -1.5707963),
        },
        skip_links=set(),
        qpos_presets={
            "rest": dict(
                zip(
                    ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_left_finger", "left_right_finger"],
                    [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.04, 0.04],
                )
            ),
            # Arm configuration actually reached near the end of a real datagen episode
            # (simple/ViperXTabletopGraspMP-v0, close+lift phase), gripper closed -- a genuine
            # grasp-adjacent pose rather than a hand-picked one.
            "grasp": dict(
                zip(
                    ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_left_finger", "left_right_finger"],
                    [-0.08261728, 0.3257859, -0.35722694, 0.09496467, 1.5119039, 1.3201247, 0.0, 0.0],
                )
            ),
        },
        # Unlike Franka/WidowX AI, viperx.xml's left_base_link sits at world x=-0.469 (inherited
        # unchanged from Aloha's own left-arm mount offset) rather than the origin -- lookat is
        # centered on the arm's own reach envelope, not world [0,0,z].
        lookat={"rest": [-0.3, 0.0, 0.25], "grasp": [-0.3, 0.0, 0.25]},
    ),
}


def _color_for_index(i: int, n: int) -> list[float]:
    """Deterministic, well-separated color for the i-th of n links, via a golden-angle HSV
    sweep (irrational step avoids any two nearby indices landing on similar hues, regardless
    of how many links n is)."""
    hue = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return [r, g, b, 1.0]


def print_legend(robot_name: str, link_names: list[str]) -> None:
    print(f"  color legend ({robot_name}):")
    for i, link_name in enumerate(link_names):
        r, g, b, _ = _color_for_index(i, len(link_names))
        print(f"    #{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}  {link_name}")


def build_spec(robot_cfg: dict) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(robot_cfg["mjcf"])

    with open(robot_cfg["spheres"]) as f:
        collision_spheres = yaml.safe_load(f)["collision_spheres"]
    link_names = sorted(k for k in collision_spheres if k not in robot_cfg["skip_links"])

    for i, link_name in enumerate(link_names):
        color = _color_for_index(i, len(link_names))
        mapping = robot_cfg["body_map"].get(link_name, link_name)
        # A tuple mapping means the curobo collision link has a nonzero fixed-joint offset from
        # its MJCF parent body (see the "viperx" body_map comment above) -- transform each
        # sphere's center out of the curobo link's own local frame into the parent body's frame
        # before parenting it there. A plain string means zero offset (the existing
        # Franka/WidowX AI behavior): the curobo link name already matches a real MJCF body, or
        # is a straight rename of one.
        if isinstance(mapping, tuple):
            body_name, offset_pos, rot_x_rad = mapping
            cos_t, sin_t = np.cos(rot_x_rad), np.sin(rot_x_rad)
            rot_x = np.array([[1, 0, 0], [0, cos_t, -sin_t], [0, sin_t, cos_t]])
        else:
            body_name, offset_pos, rot_x = mapping, None, None
        body = spec.body(body_name)
        if body is None:
            print(f"  [skip] no body named '{body_name}' (from spheres.yml link '{link_name}')")
            continue
        for sphere in collision_spheres[link_name]:
            radius = sphere["radius"]
            if radius < 0:
                continue  # curobo's convention for a disabled sphere
            center = sphere["center"]
            if offset_pos is not None:
                center = (rot_x @ np.asarray(center, dtype=float) + np.asarray(offset_pos, dtype=float)).tolist()
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0, 0],
                pos=center,
                rgba=color,
                contype=0,
                conaffinity=0,
                group=SPHERE_GROUP,
            )

    spec.worldbody.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1], diffuse=[1, 1, 1], ambient=[0.4, 0.4, 0.4], castshadow=False)
    spec.worldbody.add_light(pos=[1.0, -1.0, 1.2], dir=[-0.6, 0.6, -0.5], diffuse=[0.8, 0.8, 0.8], castshadow=False)
    spec.worldbody.add_light(pos=[-1.0, 1.0, 1.2], dir=[0.6, -0.6, -0.5], diffuse=[0.8, 0.8, 0.8], castshadow=False)
    spec.option.gravity = [0, 0, 0]
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]

    return spec


def render_robot(robot_name: str, out_dir: Path, width: int, height: int) -> None:
    robot_cfg = ROBOTS[robot_name]
    with open(robot_cfg["spheres"]) as f:
        link_names = sorted(k for k in yaml.safe_load(f)["collision_spheres"] if k not in robot_cfg["skip_links"])
    print_legend(robot_name, link_names)

    for qpos_name, qpos_values in robot_cfg["qpos_presets"].items():
        spec = build_spec(robot_cfg)
        spec.visual.global_.offwidth = width
        spec.visual.global_.offheight = height

        model = spec.compile()
        data = mujoco.MjData(model)
        for jname, val in qpos_values.items():
            data.joint(jname).qpos[0] = val
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=height, width=width)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        lookat = robot_cfg["lookat"][qpos_name]

        for view_name, (azimuth, elevation, distance) in VIEWS.items():
            cam.azimuth = azimuth
            cam.elevation = elevation
            cam.distance = distance
            cam.lookat[:] = lookat

            for label, hide_group in [("mesh", SPHERE_GROUP), ("spheres", MESH_GROUP)]:
                opt.geomgroup[:] = 1
                opt.geomgroup[hide_group] = 0
                opt.geomgroup[3] = 0  # the real physics collision primitives (invisible anyway)
                renderer.update_scene(data, camera=cam, scene_option=opt)
                img = renderer.render()
                file_path = out_dir / f"{robot_name}_{qpos_name}_{view_name}_{label}.png"
                Image.fromarray(img).save(file_path)
                print(f"wrote {file_path}")

        renderer.close()


def main(
    robot: Annotated[
        str, typer.Option(help="Render only this robot instead of both.")
    ] = "",
    out_dir: Annotated[str, typer.Option(help="Directory to write PNGs into.")] = ".render-output/collision_spheres",
    width: Annotated[int, typer.Option()] = 1280,
    height: Annotated[int, typer.Option()] = 960,
):
    if robot and robot not in ROBOTS:
        raise typer.BadParameter(f"robot must be one of {list(ROBOTS)}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for robot_name in ([robot] if robot else list(ROBOTS)):
        render_robot(robot_name, out_path, width, height)


if __name__ == "__main__":
    typer.run(main)
