# USD to MJCF Documentation

This documentation is for simulation and robotics engineers converting USD assets into simulation-ready MJCF models for MuJoCo.

The guides are organized from concepts to implementation:

1. [Overview](01-overview.md)
2. [Setup and Requirements](02-setup-and-requirements.md)
3. [Key Concepts](03-key-concepts.md)
4. [Conversion Pipeline](04-conversion-pipeline.md)
5. [Collision Development](05-collision-development.md)
6. [bin_b04 Walkthrough](06-bin-b04-walkthrough.md)
7. [Evaluating MJCF](07-evaluating-mjcf.md)
8. [Troubleshooting](08-troubleshooting.md)

## Quick start

If you want the shortest path to a working conversion, read:

1. [Setup and Requirements](02-setup-and-requirements.md)
2. [bin_b04 Walkthrough](06-bin-b04-walkthrough.md)
3. [Evaluating MJCF](07-evaluating-mjcf.md)

## Documentation scope

- Focus: USD to MJCF conversion for MuJoCo simulation.
- Focus: collision generation from visual meshes for stable contacts.
- Focus: practical evaluation with scripts in `eval/`.
- Out of scope: full SimReady ingest/authoring internals and generic USD authoring tutorials.
