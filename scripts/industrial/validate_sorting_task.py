"""Validation of g1_industrial_sorting_teleop (v2: color-coded sorting).

Stages:
  1. module imports + task/env register cleanly
  2. stub layout: _add_furniture / _add_totes place the bench, the two shelf
     variants and the red/blue totes at the expected poses; tote assets carry
     their rgba tint; the GravityShelf is gone
  3. (needs gear_sonic importable) full gym.make + env.reset: real MuJoCo scene
     has bench + left/right shelves (no GravityShelf), colored totes, screw +
     screwdrivers, robot at the canonical -x spawn (yaw 0), and compute_reward.
"""
import os
import types

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# --- 1. import + register ---------------------------------------------------
import gymnasium
import simple.envs  # noqa: F401  (registers all envs)
from simple.assets import AssetManager
from simple.core.layout import Layout
from simple.tasks.g1_industrial_sorting_teleop import (
    _BENCH_TOP,
    _BENCH_CANON_XY,
    _FURNITURE,
    _TOTES,
    _SHELF_LEFT,
    _SHELF_RIGHT,
    _TOTE_RED,
    _TOTE_BLUE,
    _OP_ROBOT_XY,
    _CANON_ROBOT_XY,
    _op_to_canon_xy,
    _op_to_canon_yaw,
    _add_furniture,
    _add_totes,
)

print("[1] import OK; env registered:",
      "simple/G1IndustrialSortingTeleop-v0" in
      [s.id for s in gymnasium.envs.registry.values()])

# --- 2. stub layout placement ------------------------------------------------
prim = AssetManager.create(
    "primitive:box", size=[0.8, 2.0, 0.1],
    position=[0.0, 0.25, _BENCH_TOP - 0.05], quaternion=[1, 0, 0, 0],
)
layout = Layout()
layout.scene = types.SimpleNamespace(table=prim)
_add_furniture(layout)
_add_totes(layout)

names = list(layout.actors.keys())
assert not any("GravityShelf" in n for n in names), "GravityShelf still present!"
print("[2] actors:", names)
for entry in _FURNITURE:
    a = layout.actors[entry["key"]]
    ex, ey = _op_to_canon_xy(*entry["op_xy"])
    ok = np.allclose(a.pose.position, [ex, ey, 0.0], atol=1e-6)
    print(f"    {entry['name']:32s} pos={[round(v,3) for v in a.pose.position]} {'OK' if ok else 'FAIL'}")
for entry in _TOTES:
    a = layout.actors[entry["key"]]
    rgba = getattr(a.asset, "rgba", None)
    print(f"    {entry['key']:12s} label={a.asset.label:14s} pos={[round(v,3) for v in a.pose.position]} rgba={rgba}")
    assert rgba is not None, f"{entry['key']} has no rgba tint"

# --- 2c. PROOF: robot<->furniture relative pose identical (operator vs canonical)
# This is the hard requirement: the scene must read as if the robot were at the
# operator pose, so the robot->furniture relative transform under the operator
# arrangement must equal the one under the mapped (canonical) arrangement.
def _rel(rx, ry, ryaw, px, py, pyaw):
    c, s = np.cos(-ryaw), np.sin(-ryaw)
    dx, dy = px - rx, py - ry
    return (c * dx - s * dy, s * dx + c * dy, (pyaw - ryaw + np.pi) % (2 * np.pi) - np.pi)

op_robot = (float(_OP_ROBOT_XY[0]), float(_OP_ROBOT_XY[1]), np.pi)
canon_robot = (float(_CANON_ROBOT_XY[0]), float(_CANON_ROBOT_XY[1]), 0.0)
rel_ok = True
for entry in _FURNITURE:
    opx, opy = entry["op_xy"]
    opyaw = entry["op_yaw"]
    cx, cy = _op_to_canon_xy(opx, opy)
    cyaw = _op_to_canon_yaw(opyaw)
    rel_op = _rel(*op_robot, opx, opy, opyaw)
    rel_canon = _rel(*canon_robot, cx, cy, cyaw)
    match = np.allclose(rel_op, rel_canon, atol=1e-9)
    rel_ok &= bool(match)
    print(f"    {entry['name']:32s} rel_op={tuple(round(v,3) for v in rel_op)} "
          f"rel_canon={tuple(round(v,3) for v in rel_canon)} {'OK' if match else 'FAIL'}")
print(f"[2c] robot<->furniture relative pose identical (op vs canonical): {rel_ok}")

# --- 3. full env (needs gear_sonic) -------------------------------------------
try:
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    cfg = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = cfg.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
except Exception as e:
    print(f"[3] SKIPPED (gear_sonic unavailable: {str(e)[:80]})")
    raise SystemExit(0)

env = gymnasium.make(
    "simple/G1IndustrialSortingTeleop-v0",
    sim_mode="mujoco", render_hz=50, physics_dt=sonic_config["SIMULATE_DT"],
    headless=True, max_episode_steps=30000, sonic_config=sonic_config,
    target=None, industrial_material=False,
)
env.reset(seed=0)
mjenv = env.unwrapped.mujoco
m, d = mjenv.mjModel, mjenv.mjData
body_names = [m.body(i).name for i in range(m.nbody)]

shelves = [n for n in body_names if "Shelving" in n]
gravity = [n for n in body_names if "GravityShelf" in n]
totes = [n for n in body_names if "bin_b04" in n]
parts = [n for n in body_names if "screw" in n or "driver" in n]
print(f"[3] scene: nbody={m.nbody} ngeom={m.ngeom}")
print("    shelves:", shelves, " gravity-shelf(should be []):", gravity)
print("    totes:", totes, " parts:", parts)

# --- 3b. part-count DR: multiple resets -> counts in [1,4], unique bodies,
#         min separation, and state_dict replay recreates the same parts -------
task = env.unwrapped.task
# NOTE: keep the number of env.reset() calls low — each one rebuilds the MuJoCo
# scene and recreates the EGL renderers, and piling them up segfaults the
# process. Part-count DR and quantity DR are therefore checked in ONE loop.
counts_ok, sep_ok, qty_ok = True, True, True
seen = []
for k in range(3):
    env.reset(seed=k + 1)
    m = env.unwrapped.mujoco.mjModel
    names_k = [m.body(i).name for i in range(m.nbody)]
    screws_l, drivers_l = task._part_labels()
    n_s, n_d = len(screws_l), len(drivers_l)
    req = task.required_counts
    seen.append((n_s, n_d, req["screws"], req["drivers"]))
    counts_ok &= 1 <= n_s <= 4 and 1 <= n_d <= 4
    counts_ok &= all(lbl in names_k for lbl in screws_l + drivers_l)
    # quantity DR: asked counts within [1, spawned] and reflected in the prompt
    qty_ok &= 1 <= req["screws"] <= n_s and 1 <= req["drivers"] <= n_d
    qty_ok &= str(req["drivers"]) in task.instruction and str(req["screws"]) in task.instruction
    # min separation among the parts actually ON the bench. Padding instances are
    # all parked together at (0, 0, -10) below the floor, so they must be excluded
    # or they'd trivially "overlap" each other.
    pts = [np.array(task.layout.actors[key].pose.position[:2])
           for key in task.layout.actors
           if (key == "target" or key.startswith("part_"))
           and task.layout.actors[key].pose.position[2] > -1.0]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < 0.08:  # tolerance under 0.10 nominal
                sep_ok = False
print(f"[3b] (spawned_s, spawned_d, asked_s, asked_d) over 3 resets: {seen}")
print(f"[3b] counts in [1,4] + bodies present: {counts_ok}  separation: {sep_ok}")
print(f"[3c] asked quantities in [1,spawned] and present in the prompt: {qty_ok}")
print(f'[3c] example prompt: "{task.instruction}"')

# --- 3d. render invariant: EVERY ObjectActor must carry `.material` ----------
# The Isaac engine reads obj_info.material for each ObjectActor. MaterialDR sets
# it while Task.reset builds the layout, so anything this task adds afterwards
# (totes, parts) must set it too or rendering dies with
# "'ObjectActor' object has no attribute 'material'".
from simple.core.actor import ObjectActor
missing = [k for k, a in task.layout.actors.items()
           if isinstance(a, ObjectActor) and not hasattr(a, "material")]
mat_ok = not missing
print(f"[3d] every ObjectActor has .material (render invariant): {mat_ok}"
      + (f"  MISSING={missing}" if missing else ""))

sd = task.state_dict()
parts_before = sorted([k for k in task.layout.actors if k.startswith("part_")])
counts_before, instr_before = task.required_counts, task.instruction
env.reset(options={"state_dict": sd})
parts_after = sorted([k for k in task.layout.actors if k.startswith("part_")])
replay_ok = parts_before == parts_after
replay_ok &= task.required_counts == counts_before and task.instruction == instr_before
print(f"[3b] replay roundtrip: {len(parts_before)} parts -> identical keys+counts+prompt: {replay_ok}")

# rebuild handles for the reward stages (the replay reset regenerated the scene)
mjenv = env.unwrapped.mujoco
m, d = mjenv.mjModel, mjenv.mjData

# tote colours on the actual geoms — expectations read from the asset definition
# (Totes_Variants) so this never goes stale when the colours are retuned.
from simple.assets.totes import Totes_Variants
for tote in (_TOTE_RED, _TOTE_BLUE):
    want = Totes_Variants[tote]["rgba"][:3]
    bid = m.body(tote).id
    gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    rgba = m.geom_rgba[gids[0]][:3]
    print(f"    {tote}: geom rgba={np.round(rgba,2).tolist()} match={np.allclose(rgba, want, atol=1e-3)}")

# robot side + yaw: canonical spawn = pelvis on -x side, body x-axis facing +x.
# .copy(): d.xpos/d.xmat are live views — stage 4 steps raw physics (no WBC),
# the robot falls, and un-copied snapshots would silently change under us.
pid = m.body("pelvis").id
px = d.xpos[pid].copy()
xaxis = d.xmat[pid].reshape(3, 3)[:, 0].copy()
print(f"    pelvis pos={np.round(px,3).tolist()}  x-axis={np.round(xaxis,2).tolist()}")
print(f"    robot on -x side: {px[0] < -0.4}   yaw 0 (facing +x): {xaxis[0] > 0.9}")

# reward callable on live state
task = env.unwrapped.task
r = task.compute_reward({}, mujoco_env=mjenv)
print(f"    compute_reward at reset = {r} (expected 0.0)")

# --- 4. reward logic: kinematic goal configuration (deterministic) ------------
# This stage tests the REWARD LOGIC, not physics (tote-on-shelf resting was
# already validated by the furniture drop tests). Bodies are posed with a ~2 mm
# penetration and mj_forward computes the contacts — no dynamic settling, so
# the check is deterministic (raw mj_step settling is flaky: the uncontrolled
# robot collapses and perturbs the scene).
import mujoco


def _place(label, x, y, z):
    d.joint(f"{label}_joint").qpos[:7] = [x, y, z, 1, 0, 0, 0]
    d.joint(f"{label}_joint").qvel[:6] = 0


_DECK_Z = 0.98  # a validated cart shelf level; tote origin sits at its base


def _pose_goal(red_shelf_xy, blue_shelf_xy, n_screws=None, n_drivers=None):
    """Pose the goal: totes on their shelves with the asked-for parts inside.

    Places `required_counts` of each class (or an override, to test the
    threshold) spread slightly so they don't perfectly overlap but stay well
    within the in-tote XY tolerance.
    """
    screws_l, drivers_l = task._part_labels()
    req = task.required_counts
    n_s = req["screws"] if n_screws is None else n_screws
    n_d = req["drivers"] if n_drivers is None else n_drivers
    _place(_TOTE_RED, red_shelf_xy[0], red_shelf_xy[1], _DECK_Z - 0.002)
    _place(_TOTE_BLUE, blue_shelf_xy[0], blue_shelf_xy[1], _DECK_Z - 0.002)
    # Parts beyond the requested count are PARKED far away — otherwise they'd
    # linger inside the tote from a previous _pose_goal call and the "one short"
    # check would still see them.
    for i, lbl in enumerate(screws_l):
        if i < n_s:
            _place(lbl, red_shelf_xy[0] + 0.02 * i, red_shelf_xy[1], _DECK_Z + 0.01)
        else:
            _place(lbl, 4.0 + 0.1 * i, 4.0, 1.0)
    for i, lbl in enumerate(drivers_l):
        if i < n_d:
            _place(lbl, blue_shelf_xy[0] + 0.02 * i, blue_shelf_xy[1], _DECK_Z + 0.01)
        else:
            _place(lbl, -4.0 - 0.1 * i, 4.0, 1.0)
    mujoco.mj_forward(m, d)


# read the actual placed shelf centers (robust to the operator->canonical map)
LEFT_XY = tuple(float(v) for v in d.xpos[m.body(_SHELF_LEFT).id][:2])
RIGHT_XY = tuple(float(v) for v in d.xpos[m.body(_SHELF_RIGHT).id][:2])
_pose_goal(LEFT_XY, RIGHT_XY)
r_goal = task.compute_reward({}, mujoco_env=mjenv)
print(f"[4] reward in goal configuration = {r_goal} (expected 1.0)")

# cross-check: swap the totes (red on RIGHT shelf, blue on LEFT) -> shelf
# credit must vanish (only the two in-tote conditions remain).
_pose_goal(RIGHT_XY, LEFT_XY)
r_swap = task.compute_reward({}, mujoco_env=mjenv)
print(f"[4] reward with swapped shelves = {r_swap} (expected 0.5, no shelf credit)")

# --- 4b. quantity threshold, deterministic ------------------------------------
# Force the asked counts to the number of parts actually on the bench, then pose
# the goal with exactly that many (must be 1.0) and with one screw fewer (must
# lose exactly the screw condition -> 0.75). Independent of the episode's draw.
screws_l, drivers_l = task._part_labels()
n_s, n_d = len(screws_l), len(drivers_l)
task._required_counts = {"screws": n_s, "drivers": n_d}
print(f"[4b] forcing asked = visible: screws={n_s} drivers={n_d}")

_pose_goal(LEFT_XY, RIGHT_XY, n_screws=n_s, n_drivers=n_d)
r_exact = task.compute_reward({}, mujoco_env=mjenv)
print(f"[4b] reward with exactly the asked quantity = {r_exact} (expected 1.0)")

_pose_goal(LEFT_XY, RIGHT_XY, n_screws=max(0, n_s - 1), n_drivers=n_d)
r_short = task.compute_reward({}, mujoco_env=mjenv)
print(f"[4b] reward with ONE SCREW SHORT = {r_short} (expected 0.75, screw condition lost)")

ok = (
    rel_ok and qty_ok and replay_ok
    and _SHELF_LEFT in body_names and _SHELF_RIGHT in body_names
    and not gravity and _TOTE_RED in body_names and _TOTE_BLUE in body_names
    and px[0] < -0.4 and xaxis[0] > 0.9 and r == 0.0
    and r_goal == 1.0 and r_swap <= 0.5
    and r_exact == 1.0 and r_short == 0.75
    and mat_ok
)
print("[done] FULL VALIDATION:", "PASS" if ok else "FAIL")
env.close()
