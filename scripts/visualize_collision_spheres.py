"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Visualize CuRobo self-collision spheres (spheres.yml) overlaid on the real robot mesh, for
Franka FR3 and WidowX AI side by side in one MuJoCo viewer.

WHY
---
`docs/teleop_simple_study/widowx_ai_integration_status.md` (section 12+) investigates a
self-collision false-positive bug on WidowX AI: curobo's `spheres.yml` (auto-fit against the
robot's detailed STL meshes) over-approximates the true collision volume for several links,
which the mesh's own hand-tuned MuJoCo collision primitives (cylinder/capsule/box) don't. All of
that was diagnosed numerically (self-collision cost values, per-sphere-pair gap distances) --
this script is for looking at the actual geometry instead of reading numbers: it drops every
sphere from spheres.yml into the MuJoCo scene, as a translucent sphere geom parented to the same
body the real mesh geom belongs to, so moving/rotating in the viewer shows exactly how much each
sphere overshoots (or undershoots) the link it's supposed to approximate. Franka is included
alongside WidowX for comparison -- Franka's self-collision checking isn't broken (see the status
doc), so its spheres are a visual reference for "what a reasonably tight fit looks like".

HOW SPHERES ARE PLACED
-----------------------
`spheres.yml`'s sphere centers are in each link's own LOCAL body frame -- the same frame
`MjSpec.body(name).add_geom(pos=...)` uses for a geom parented to that body. So no FK/transform
math is needed here: each sphere is added as a literal child geom of the matching MuJoCo body,
`contype=0`/`conaffinity=0` (purely visual, never affects physics) and translucent (alpha ~0.35)
so the real mesh stays visible underneath. Because they're real child geoms (not one-off markers
computed at a fixed qpos), dragging a joint in the viewer moves its spheres with it via MuJoCo's
own forward kinematics -- exactly like the real collision geoms would.

Two name mismatches needed explicit mapping (BODY_NAME_MAP below), found by direct inspection,
not assumed:
- Franka: spheres.yml (curobo's bundled NVIDIA config) uses `fr3_link0`..`fr3_link7`,
  `fr3_hand`, `fr3_leftfinger`, `fr3_rightfinger`; `panda.xml` bodies are named `link0`..`link7`,
  `hand`, `left_finger`, `right_finger` (no `fr3_` prefix, underscore in the finger names).
- WidowX AI: `gripper_left`/`gripper_right` are separate links in the URDF (and so have their
  own sphere entries in spheres.yml), but `wxai_follower.xml` never gives them a separate body --
  their mesh is drawn directly inside `carriage_left`/`carriage_right`. Checked the URDF's fixed
  joint between them (`<joint name="left_gripper_joint" type="fixed"><origin xyz="0 0 0"/>`) --
  zero offset, so gripper_left/right spheres can be parented straight onto
  carriage_left/right with no coordinate correction.

USAGE
-----
    .venv/bin/python scripts/visualize_collision_spheres.py

Options:
    --widowx-qpos rest|close-high   Which WidowX AI joint configuration to display (default: rest,
                                     the pose section 12's investigation centers on).
    --alpha 0.35                    Sphere transparency (0=invisible, 1=opaque).

In the viewer: number keys 2/4 toggle MuJoCo geom groups 2 (real meshes) / 4 (collision spheres)
on and off independently, so you can look at spheres alone or meshes alone.
"""

from __future__ import annotations

from typing import Annotated

import mujoco
import mujoco.viewer
import numpy as np
import typer
import yaml

FRANKA_MJCF = "data/robots/franka_fr3/panda.xml"
FRANKA_SPHERES = "data/robots/franka_fr3/spheres.yml"
FRANKA_ARM_QPOS = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]  # franka_fr3.py's init_joint_states
FRANKA_GRIPPER_QPOS = [0.04, 0.04]

WIDOWX_MJCF = "data/robots/widowx_ai/wxai_follower.xml"
WIDOWX_SPHERES = "data/robots/widowx_ai/spheres.yml"
WIDOWX_QPOS_PRESETS = {
    "rest": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # scripts/test_widowx_ai_curobo_ik.py's "close-high" waypoint, solved qpos (see that script's
    # WAYPOINTS docstring for provenance).
    "close-high": [0.0, 0.13631485402584076, 0.8843041062355042, -0.7479904890060425, 0.0, 0.0],
}

SPHERE_GROUP = 4  # NOT 3 -- both panda.xml and wxai_follower.xml already use group 3 for their
# own real (invisible-by-default) "collision" class primitives; sharing it would make our
# visualization spheres untoggleable independently of those (confirmed by grepping both MJCFs).

# link_name (as it appears in spheres.yml) -> body name (as it appears in the robot's own MJCF).
# Identity when absent from the map.
BODY_NAME_MAP = {
    "franka": {
        **{f"fr3_link{i}": f"link{i}" for i in range(8)},
        "fr3_hand": "hand",
        "fr3_leftfinger": "left_finger",
        "fr3_rightfinger": "right_finger",
        # "attached_object" in franka's spheres.yml isn't a robot link (a placeholder curobo uses
        # for objects grasped mid-episode) -- skipped explicitly, see add_spheres() below.
    },
    "widowx_ai": {
        "gripper_left": "carriage_left",
        "gripper_right": "carriage_right",
    },
}


def add_spheres(spec: mujoco.MjSpec, spheres_path: str, name_map: dict[str, str], rgba: list[float]) -> None:
    with open(spheres_path) as f:
        collision_spheres = yaml.safe_load(f)["collision_spheres"]

    for link_name, spheres in collision_spheres.items():
        if link_name == "attached_object":
            continue
        body_name = name_map.get(link_name, link_name)
        body = spec.body(body_name)
        if body is None:
            print(f"  [skip] no body named '{body_name}' (from spheres.yml link '{link_name}')")
            continue
        for sphere in spheres:
            radius = sphere["radius"]
            if radius < 0:
                continue  # curobo's convention for a disabled sphere
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0, 0],
                pos=sphere["center"],
                rgba=rgba,
                contype=0,
                conaffinity=0,
                group=SPHERE_GROUP,
            )


def main(
    widowx_qpos: Annotated[
        str, typer.Option(help="Which WidowX AI joint configuration to display.")
    ] = "rest",
    alpha: Annotated[float, typer.Option(help="Sphere transparency (0=invisible, 1=opaque).")] = 0.35,
):
    if widowx_qpos not in WIDOWX_QPOS_PRESETS:
        raise typer.BadParameter(f"widowx_qpos must be one of {list(WIDOWX_QPOS_PRESETS)}")

    print("Loading Franka FR3 ...")
    franka_spec = mujoco.MjSpec.from_file(FRANKA_MJCF)
    add_spheres(franka_spec, FRANKA_SPHERES, BODY_NAME_MAP["franka"], rgba=[0.2, 0.4, 1.0, alpha])

    print("Loading WidowX AI ...")
    widowx_spec = mujoco.MjSpec.from_file(WIDOWX_MJCF)
    add_spheres(widowx_spec, WIDOWX_SPHERES, BODY_NAME_MAP["widowx_ai"], rgba=[1.0, 0.2, 0.2, alpha])

    scene = mujoco.MjSpec()
    scene.option.gravity = [0, 0, 0]  # static kinematic display -- no controller is driving
    # either robot's actuators, so gravity would just make both arms sag away from the qpos
    # we're about to set, defeating the point of a side-by-side visual comparison.

    # Franka's own panda.xml bundles a `<light name="top" .../>`, but WidowX's wxai_follower.xml
    # defines none -- a bare `MjSpec()` has no implicit light either, so without this the scene
    # renders essentially black. Same light engines/mujoco.py's `_setup_scene()` always adds for
    # exactly this reason.
    scene.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], castshadow=False)

    franka_frame = scene.worldbody.add_frame(pos=[-0.6, 0, 0], quat=[1, 0, 0, 0])
    scene.attach(franka_spec, frame=franka_frame)
    widowx_frame = scene.worldbody.add_frame(pos=[0.6, 0, 0], quat=[1, 0, 0, 0])
    scene.attach(widowx_spec, frame=widowx_frame)

    model = scene.compile()
    data = mujoco.MjData(model)

    for jname, val in zip([f"joint{i}" for i in range(1, 8)], FRANKA_ARM_QPOS):
        data.joint(jname).qpos[0] = val
    for jname, val in zip(["finger_joint1", "finger_joint2"], FRANKA_GRIPPER_QPOS):
        data.joint(jname).qpos[0] = val

    for jname, val in zip([f"joint_{i}" for i in range(6)], WIDOWX_QPOS_PRESETS[widowx_qpos]):
        data.joint(jname).qpos[0] = val
    # right_carriage_joint mimics left_carriage_joint exactly (MuJoCo <equality>,
    # polycoef="0 1 0 0 0") but mj_forward alone never resolves <equality> constraints (only
    # mj_step's constraint solver does, which this viewer's interactive loop does call
    # repeatedly -- but the FIRST frame before any stepping needs it set explicitly too, or the
    # right finger visibly starts closed while the left one is open).
    data.joint("left_carriage_joint").qpos[0] = 0.044  # open
    data.joint("right_carriage_joint").qpos[0] = 0.044

    mujoco.mj_forward(model, data)

    print(f"\nFranka FR3 on the left (blue spheres), WidowX AI on the right (red spheres, "
          f"qpos={widowx_qpos!r}).")
    print("Viewer keys: '2' toggles the real meshes, '4' toggles the collision spheres.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.3]
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        while viewer.is_running():
            # Kinematic-only display: no mj_step (no gravity/actuators to integrate against),
            # just keep the viewer responsive to camera/UI interaction and any manual joint drags.
            mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    typer.run(main)
