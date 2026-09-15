# Setup and Requirements

This page covers environment setup for both conversion and evaluation.

## Conversion environment

From `requirements.txt`, the converter depends on:

- `numpy`
- `scipy`
- `usd-core>=25.5`
- `trimesh`
- `coacd`

Recommended setup:

```bash
conda create --name usd2mjcf python=3.10
conda activate usd2mjcf
pip install -r requirements.txt
```

## Evaluation environment

Evaluation scripts in `eval/` use MuJoCo and video rendering dependencies.

```bash
pip install -r eval/requirements.txt
```

`eval/requirements.txt` includes:

- `mujoco>=3.0`
- `imageio>=2.30`
- `imageio-ffmpeg`

## Headless rendering

When running on servers or CI without a display:

```bash
export MUJOCO_GL=egl
```

If EGL is unavailable, try:

```bash
export MUJOCO_GL=osmesa
```

## Input asset requirements

For practical conversion:

- Use `.usd` or `.usda` input files.
- Keep referenced external assets reachable (or resolvable to local cache).
- Prefer text `.usda` during debugging because references are easier to inspect.

For the sample walkthrough in this repository, use:

- `assets/bin_b04.usda` as the primary input.

## Output and storage expectations

- Converter outputs are written under `{output_path}/MJCF/`.
- Collision generation can emit many OBJ files (`bin_b04` generates 114 collision OBJ parts).
- External references may be mirrored to `assets/_resolved_cache/` unless a custom cache path is provided.

## Network and resolver notes

For assets that reference remote Omniverse content:

- Keep `--resolve_external_assets` enabled (default behavior).
- Use `--resolver_strict` when you want conversion to fail on unresolved references.
- Use `--asset_cache_dir` to control where mirrored dependencies are stored.

## Optional debug tooling

- `lightwheel/srl/from_usd/usd_to_graphviz_cli.py` can help inspect the transform graph structure.
- `eval/drop_test.py --viewer` is useful for fast visual sanity checks before generating videos.
