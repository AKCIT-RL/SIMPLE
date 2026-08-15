"""Validate industrial parts: load via AssetManager + dynamic MuJoCo drop test.

Builds the free rigid body the way the SIMPLE MuJoCo engine does (convex hull
geoms + freejoint, total mass ~0.1 kg), initialised at each stable pose, and
checks it rests stably (no tunneling, bounded contacts, finite velocity).
"""
import os, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "egl")
from simple.assets import AssetManager

def build_xml(asset, stable_pose, drop=0.15):
    hulls = asset.collision_meshes_mujoco
    n = len(hulls)
    meshes = "\n".join(f'<mesh name="m{i}" file="{h}"/>' for i,h in enumerate(hulls))
    geoms  = "\n".join(f'<geom type="mesh" mesh="m{i}" mass="{0.1/n:.5f}" condim="4" friction="1 0.05 0.001"/>' for i in range(n))
    z = float(stable_pose[2]) + drop
    qw,qx,qy,qz = (float(v) for v in stable_pose[3:7])
    return f"""<mujoco><compiler angle="radian"/><option timestep="0.004" gravity="0 0 -9.81"/>
      <asset>{meshes}</asset>
      <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
        <body name="part" pos="0 0 {z:.4f}" quat="{qw} {qx} {qy} {qz}"><freejoint/>
{geoms}
        </body></worldbody></mujoco>"""

def validate(res_id, asset_id):
    mgr = AssetManager.get(res_id); asset = mgr.load(asset_id)
    print(f"[{res_id}:{asset_id}] name={asset.name} hulls={len(asset.collision_meshes_mujoco)} "
          f"stable_poses={len(asset.stable_poses)} usd={os.path.basename(asset.usd_path)}")
    ok_all=True
    for k,sp in enumerate(asset.stable_poses):
        m = mujoco.MjModel.from_xml_string(build_xml(asset, sp)); d = mujoco.MjData(m)
        qadr = m.jnt_qposadr[m.body("part").jntadr[0]]
        z0=d.qpos[qadr+2]; maxcon=0; nan=False
        for _ in range(500):
            mujoco.mj_step(m,d); maxcon=max(maxcon,d.ncon)
            if not np.isfinite(d.qvel).all(): nan=True; break
        z=d.qpos[qadr+2]; v=float(np.linalg.norm(d.qvel[:3]))
        tunneled = z < -0.1
        ok = (not tunneled) and (not nan) and v<0.06 and maxcon<200
        ok_all &= ok
        print(f"    pose[{k}] z:{z0:.3f}->{z:.3f} |v|={v:.4f} max_ncon={maxcon} "
              f"tunnel={tunneled} nan={nan}  {'OK' if ok else 'FAIL'}")
    print(f"    => {'PASS' if ok_all else 'FAIL'}")
    return ok_all

if __name__=="__main__":
    r={}
    for a in ("factory_gear_large","t_connector_physics"):
        r[a]=validate("industrial_parts", a)
    print("\nSUMMARY:", {k:('PASS' if v else 'FAIL') for k,v in r.items()})
