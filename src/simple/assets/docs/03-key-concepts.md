# Key Concepts

This page defines the mental model behind the converter so command output is easier to reason about.

## Mental model of the process

A useful way to think about USD-to-MJCF in this repository:

1. **Interpret USD structure**: collect bodies, transforms, joints, mesh prims, and material bindings.
2. **Build MJCF skeleton**: write bodies/joints/geoms and export visual OBJ meshes.
3. **Add collision for simulation**: decompose visuals into convex pieces and inject collision geoms.
4. **Validate behavior**: run drop/stack simulations and inspect contact behavior.

The first two steps produce a model that can load. The third step makes it physically reliable for contact-rich simulation.

## USD side concepts

### Scene and kinematics

- USD prim hierarchy (Xforms + meshes) defines spatial structure.
- `UsdPhysics` joints define articulation.
- The converter builds an intermediate transform graph before writing MJCF.

### Materials

- USD material bindings are interpreted and translated to MJCF material/texture entries.
- Practical output in MJCF focuses on diffuse color/texture behavior.

### Collision in USD

- USD assets may contain collision-authoring primitives/API tags.
- In this converter pipeline, collision-tagged USD prims are not directly exported as MJCF collision geoms.
- Collision is generated later from visual meshes using convex decomposition.

## MJCF side concepts

### Geom class roles

The converter uses MJCF defaults to separate render and contact behavior:

- `class="visual"` geoms are non-colliding (`conaffinity=0`, `contype=0`).
- `class="collision"` geoms are used for contact simulation.

This separation is why a model can look correct but still fail physically if collision geoms are missing.

### Mesh path contract

- `<mesh file="...">` paths are relative to the XML location.
- `visuals/` and `collision/` folders must remain alongside the XML unless paths are updated.

### Root body and simulation

- Root names can vary (`Root`, `RootNode`, or others).
- Eval scripts auto-detect common roots, but explicit `--root-body` is available when needed.

## Transform graph reduction (why it exists)

USD scenes can contain structures that do not map directly to MJCF expectations. The reduction step:

- trims to an MJCF-compatible tree,
- aligns joint/link frame assumptions,
- and prepares data for deterministic XML export.

For static props, this often collapses to a single root body with visual/collision mesh geoms.

## Practical concept: skeleton vs simulation-ready output

It is common to see two states of output:

- **Skeleton output**: XML/body structure with no usable collision geoms.
- **Simulation-ready output**: visual geoms plus generated convex collision geoms.

For `bin_b04`, the complete reference output is at `assets/bin_b04/MJCF/bin_b04.xml` and includes one visual mesh plus many collision parts.

## Why convex decomposition is central

MuJoCo contact simulation is sensitive to collision geometry quality.

- Fewer, overly coarse hulls can lead to penetration and unstable stacking.
- Too many hulls improve fit but can increase simulation cost.

The docs and eval workflow therefore center on tuning this trade-off.
