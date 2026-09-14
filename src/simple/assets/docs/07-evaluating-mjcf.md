# Evaluating MJCF

After conversion, use the scripts in `eval/` to validate that the MJCF behaves plausibly in MuJoCo.

## Evaluation goals

- Verify floor contact and settling behavior.
- Verify object-to-object contact during stacking.
- Produce visual artifacts (`.mp4` or `.gif`) for review.
- Collect simple runtime metrics for quick triage.

## Prerequisites

1. Generate collision-enabled MJCF:

```bash
python test/usd2mjcf_test.py assets/bin_b04.usda --generate_collision
```

2. Install eval dependencies:

```bash
pip install -r eval/requirements.txt
```

3. If headless:

```bash
export MUJOCO_GL=egl
```

## Script 1: Drop test

Run:

```bash
python eval/drop_test.py --xml assets/bin_b04/MJCF/bin_b04.xml
```

What it does:

- Loads MJCF.
- Resolves a root body.
- Ensures a free joint exists for drop behavior.
- Adds a floor plane at runtime.
- Simulates and writes a video.
- Prints metrics such as first contact time and final height.

Useful variants:

```bash
python eval/drop_test.py --xml assets/bin_b04/MJCF/bin_b04.xml --viewer
python eval/drop_test.py --xml assets/bin_b04/MJCF/bin_b04.xml --height 2.0 --duration 5.0
python eval/drop_test.py --xml assets/bin_b04/MJCF/bin_b04.xml --output eval/output/bin_b04_drop.gif
```

Interpretation:

- Good sign: finite contact time, stable rest pose.
- Warning sign: `First contact time: none`, floor penetration, extreme jitter.

## Script 2: Stack spawn test

Run:

```bash
python eval/stack_spawn_test.py \
  --xml assets/bin_b04/MJCF/bin_b04.xml \
  --count 4 \
  --spawn-interval 1.5 \
  --rotation-step 15 \
  --tilt-step 5 \
  --duration 12 \
  --output eval/output/bin_b04_stack.gif
```

What it does:

- Spawns multiple copies of the model over time.
- Applies incremental yaw/pitch rotations.
- Exercises object-to-object contacts and stability.

Interpretation:

- Good sign: objects stack and settle without violent instability.
- Warning sign: interpenetration, explosive separation, or pass-through behavior.

## Recommended eval sequence for new assets

```bash
python eval/drop_test.py --xml path/to/model.xml --viewer
python eval/drop_test.py --xml path/to/model.xml --output eval/output/my_asset_drop.gif
python eval/stack_spawn_test.py --xml path/to/model.xml --count 4 --output eval/output/my_asset_stack.gif
```

## Important caveat

These scripts are simulation smoke tests, not strict pass/fail benchmarks.

- There are no built-in assertion thresholds for metrics.
- Success means the script runs, emits outputs, and the resulting behavior is physically plausible under review.

For complete flag documentation, see `eval/README.md`.
