"""Validate an extracted furniture MJCF in MuJoCo (Plan-A collision integrity).

Drops a tote-sized box above the piece and checks: model compiles, no tunneling
through the floor, bounded contact count (no explosion), finite/stable velocity,
and reports the settle height. Prints a per-piece A/B verdict hint.
"""
import argparse, os, numpy as np, mujoco

def validate(name, base_dir, drop_extra=0.2, steps=600):
    mjcf = os.path.abspath(os.path.join(base_dir, name, f"{name}.mjcf.xml"))
    # approx top from visual bbox
    import trimesh
    vis = trimesh.load(os.path.join(base_dir, name, "visuals", f"{name}.obj"), process=False)
    top = float(vis.bounds[1][2]); zmin = float(vis.bounds[0][2])
    xml = f"""<mujoco>
      <compiler meshdir="{os.path.abspath(os.path.join(base_dir,name))}" angle="radian"/>
      <option timestep="0.005" gravity="0 0 -9.81"/>
      <include file="{mjcf}"/>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1"/>
        <body name="tote" pos="0 0 {top+drop_extra:.3f}"><freejoint/>
          <geom type="box" size="0.08 0.08 0.06" density="200"/></body>
      </worldbody></mujoco>"""
    p = f"/tmp/_val_{name}.xml"; open(p,"w").write(xml)
    m = mujoco.MjModel.from_xml_path(p); d = mujoco.MjData(m)
    qadr = m.jnt_qposadr[m.body("tote").jntadr[0]]
    maxcon=0; tunneled=False; nan=False
    for _ in range(steps):
        mujoco.mj_step(m,d)
        maxcon=max(maxcon,d.ncon)
        z=d.qpos[qadr+2]
        if z < zmin-0.15: tunneled=True
        if not np.isfinite(d.qvel).all(): nan=True; break
    z=d.qpos[qadr+2]; v=float(np.linalg.norm(d.qvel[:3]))
    ok = (not tunneled) and (not nan) and np.isfinite(v) and maxcon<200
    print(f"[{name}]")
    print(f"   ngeom={m.ngeom} top={top:.3f} zmin={zmin:.3f}")
    print(f"   settle_z={z:.3f}  |v|={v:.4f}  max_ncon={maxcon}  tunneled={tunneled}  nan={nan}")
    print(f"   VERDICT: {'PASS (Plan A ok)' if ok else 'FAIL -> candidate for Plan B'}")
    return ok

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--base",default="data/assets/industrial")
    ap.add_argument("--names",nargs="+",required=True); a=ap.parse_args()
    res={n:validate(n,a.base) for n in a.names}
    print("\nSUMMARY:", {k:('PASS' if v else 'FAIL') for k,v in res.items()})
