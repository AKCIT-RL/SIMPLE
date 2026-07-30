# Troubleshooting

Use this page to diagnose common USD-to-MJCF conversion and evaluation failures.

## Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Eval errors with `0 geoms` | Conversion output is incomplete or collision generation was skipped | Re-run conversion with `--generate_collision` and inspect XML for geoms |
| `Unresolved USD references` during conversion | Remote/file references could not be mirrored or rewritten | Keep resolver enabled, verify connectivity/paths, try `--resolver_strict` to surface exact failures |
| `First contact time: none` in drop test | Collision geoms missing or ineffective | Confirm `class="collision"` geoms exist and regenerate with higher resolution |
| Object falls through floor | Collision decomposition too coarse or invalid | Increase `--resolution`, regenerate, and retest |
| Severe jitter or explosive motion | Poor collision hull quality or unstable contact configuration | Regenerate with tuned decomposition, inspect hull count/shape, reduce complexity where possible |
| Mesh file not found when loading XML | XML moved without `visuals/` / `collision/` folders | Keep mesh directories relative to XML or update mesh paths |
| Renderer/runtime errors in headless mode | MuJoCo backend not set for offscreen rendering | Set `MUJOCO_GL=egl` (or `osmesa`) before running eval |
| Wrong root body chosen in eval | Root naming differs from defaults | Pass `--root-body <name>` explicitly |

## Quick diagnostic commands

Check collision geoms:

```bash
rg "class=\"collision\"" path/to/model.xml
```

Check referenced mesh files in XML:

```bash
rg "<mesh file=" path/to/model.xml
```

Count generated collision OBJ files:

```bash
ls path/to/MJCF/collision | wc -l
```

## Minimal recovery workflow

```bash
python test/usd2mjcf_test.py assets/bin_b04.usda --generate_collision
python eval/drop_test.py --xml assets/bin_b04/MJCF/bin_b04.xml --viewer
python eval/stack_spawn_test.py --xml assets/bin_b04/MJCF/bin_b04.xml --count 3
```

If behavior is still unstable, raise decomposition resolution and rerun evaluation.
