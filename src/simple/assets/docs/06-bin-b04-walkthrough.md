# bin_b04 Walkthrough

This walkthrough demonstrates the full USD-to-MJCF flow using `bin_b04`.

## Inputs and reference paths

Primary USD input:

- `assets/bin_b04.usda`

Reference complete bundle:

- `assets/bin_b04/`

Key reference output:

- `assets/bin_b04/MJCF/bin_b04.xml`

## Step 1: Prepare environment

```bash
conda create --name usd2mjcf python=3.10
conda activate usd2mjcf
pip install -r requirements.txt
```

## Step 2: Convert with collision generation

```bash
python test/usd2mjcf_test.py assets/bin_b04.usda --generate_collision
```

What to expect:

- Resolver summary logs (`downloaded`, `reused`, `rewritten`, `unresolved`).
- MJCF output under `assets/MJCF/`.
- `visuals/` and `collision/` output folders next to the generated XML.

## Step 3: Verify generated structure

```bash
ls assets/MJCF
ls assets/MJCF/visuals
ls assets/MJCF/collision | wc -l
```

Checks:

- `assets/MJCF/bin_b04.xml` exists.
- At least one visual OBJ exists.
- Collision directory contains convex parts.

## Step 4: Confirm collision geoms in XML

```bash
rg "class=\"collision\"" assets/MJCF/bin_b04.xml
```

If there are no matches, the output is not simulation-ready for contact tests.

## Step 5: Compare with reference complete model

```bash
rg -c "class=\"collision\"" assets/MJCF/bin_b04.xml
rg -c "class=\"collision\"" assets/bin_b04/MJCF/bin_b04.xml
```

The exact numbers may differ with decomposition settings, but both should contain collision geoms.

## Optional: higher-fidelity collision run

```bash
python test/usd2mjcf_test.py assets/bin_b04.usda \
  --generate_collision \
  --preprocess_resolution=40 \
  --resolution=4000
```

Higher settings can improve contact fidelity but generally increase decomposition time and hull count.

## Notes on asset metadata

`assets/bin_b04/bin_b04.json` is ingest/pipeline metadata from upstream content preparation.  
It is useful context, but it is not used as runtime converter config in `test/usd2mjcf_test.py`.

## Next step

Proceed to [Evaluating MJCF](07-evaluating-mjcf.md) to validate runtime behavior in MuJoCo.
