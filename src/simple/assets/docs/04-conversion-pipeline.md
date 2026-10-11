# Conversion Pipeline

This page explains the concrete execution path from USD input to MJCF output in this repository.

## Entry point

Use:

```bash
python test/usd2mjcf_test.py <input_path> [options]
```

Main stages:

1. Resolve external references (optional, enabled by default).
2. Convert USD scene structure and visuals into MJCF.
3. Generate convex collision meshes (optional, via `--generate_collision`).

## Stage 1: Asset resolution

Module: `utils/resolve_usd_assets.py`  
Called from: `test/usd2mjcf_test.py`

What it does:

- Parses text USD references of the form `@...@`.
- Mirrors remote dependencies (for example HTTPS payloads) into local cache.
- Rewrites referenced paths to local cache copies when needed.
- Optionally errors on unresolved references in strict mode.

Relevant flags:

- `--resolve_external_assets` / `--no-resolve_external_assets`
- `--asset_cache_dir`
- `--resolver_strict`

Typical resolver log summary:

```text
[resolver] downloaded=<n> reused=<n> rewritten=<n> unresolved=<n>
```

## Stage 2: Core USD -> MJCF conversion

Modules:

- `test/usd2mjcf_test.py`
- `lightwheel/srl/from_usd/to_mjcf.py`
- `lightwheel/srl/from_usd/transform_graph.py`
- `lightwheel/srl/from_usd/transform_graph_tools.py`

What happens:

1. Load USD stage and initialize `UsdToMjcf`.
2. Build a transform graph from USD prims and physics joints.
3. Reduce/normalize graph for MJCF compatibility.
4. Export XML with bodies, joints, and visual geoms.
5. Export visual OBJ meshes and material resources.

Default output location:

- If `--output_path` is omitted, output root is the input file's parent directory.
- MJCF output is written under `{output_root}/MJCF/`.

For `assets/bin_b04.usda`, default target becomes:

```text
assets/MJCF/
```

## Stage 3: Collision post-processing

Only runs when:

```bash
--generate_collision
```

Modules:

- `utils/add_collision.py`
- `utils/format_coacd.py`

What it does:

1. Parse generated MJCF XML.
2. Find visual mesh geoms.
3. Load matching visual OBJ meshes.
4. Run CoACD convex decomposition.
5. Write convex parts to `collision/*.obj`.
6. Patch MJCF with collision `<mesh>` assets and `class="collision"` geoms.

Tuning flags:

- `--preprocess_resolution` (default `20`)
- `--resolution` (default `2000`)

## CLI parameters quick reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_path` | required | none | USD or USDA input file |
| `--output_path` | optional | input parent | Output root; converter writes to `MJCF/` under this path |
| `--generate_collision` | flag | off | Generate convex collision meshes and patch MJCF |
| `--preprocess_resolution` | optional int | `20` | CoACD preprocessing resolution |
| `--resolution` | optional int | `2000` | CoACD decomposition resolution |
| `--resolve_external_assets` / `--no-resolve_external_assets` | optional bool | on | Enable/disable reference resolver |
| `--asset_cache_dir` | optional path | `assets/_resolved_cache` | Resolver cache directory |
| `--resolver_strict` | flag | off | Fail conversion on unresolved refs |

## Output directory contract

```text
{output_path}/MJCF/
├── {asset_name}.xml
├── visuals/
│   ├── {mesh}.obj
│   └── textures/               # if textures are exported
└── collision/                  # only when --generate_collision
    └── {mesh}001.obj ...
```

## Minimal command patterns

Visual-only conversion:

```bash
python test/usd2mjcf_test.py assets/bin_b04.usda
```

Simulation-ready conversion:

```bash
python test/usd2mjcf_test.py assets/bin_b04.usda --generate_collision
```

Strict resolver with custom cache:

```bash
python test/usd2mjcf_test.py assets/bin_b04.usda \
  --generate_collision \
  --asset_cache_dir /tmp/usd_cache \
  --resolver_strict
```
