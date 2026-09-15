# Overview

USD2MJCF converts USD assets into MJCF so they can run in MuJoCo physics simulation.

## What "usable MJCF" means

A usable MJCF in this repository has:

- A loadable XML model in `MJCF/`.
- Mesh assets referenced by relative path from the XML (`visuals/`, and optionally `collision/`).
- A root body that can be simulated in MuJoCo.
- Collision geoms (`class="collision"`) when physics contact behavior is required.

Without collision generation, conversion can still produce a valid XML skeleton, but it is often not physically useful for contact-heavy simulation.

## Mental model: two-phase conversion

Think of the process in two phases:

1. **Structure and visuals**: read USD, build an MJCF body/geom graph, export visual meshes and materials.
2. **Collision synthesis**: decompose visual meshes into convex parts and inject collision geoms into MJCF.

This distinction matters because USD collision authoring and MJCF collision requirements are different.

## Mental model: USD collision vs MJCF collision

- In USD/Omniverse workflows, collision data can be authored as dedicated collision prims.
- In this converter, collision-tagged USD prims are filtered out during main conversion.
- MuJoCo collision meshes must be convex for robust contact simulation.
- The repository handles this by generating convex collision hulls from visual OBJ meshes in a post-process step (`--generate_collision`).

## End-to-end pipeline

```mermaid
flowchart LR
    usdInput[USD_input] --> resolver[resolve_usd_input]
    resolver --> converter[UsdToMjcf]
    converter --> visualMJCF[Visual_MJCF_plus_OBJs]
    visualMJCF -->|"--generate_collision"| coacd[CoACD_decomposition]
    coacd --> simReady[Simulation_ready_MJCF]
    simReady --> evalScripts[eval_drop_and_stack]
```

## Typical outputs

```text
{output_path}/MJCF/
├── {asset_name}.xml
├── visuals/
│   ├── {mesh}.obj
│   └── textures/              # when material textures are exported
└── collision/                 # only with --generate_collision
    └── {mesh}001.obj ...
```

## Recommended path through these docs

1. [Setup and Requirements](02-setup-and-requirements.md)
2. [Key Concepts](03-key-concepts.md)
3. [Conversion Pipeline](04-conversion-pipeline.md)
4. [Collision Development](05-collision-development.md)
5. [bin_b04 Walkthrough](06-bin-b04-walkthrough.md)
6. [Evaluating MJCF](07-evaluating-mjcf.md)
