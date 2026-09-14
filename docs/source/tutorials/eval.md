# Policy Evaluation

Benchmark tasks for $\Psi_0$

| Tasks | Whole-body | Motion Planning | Teleoperation |
|----------|----------|----------|----------| 
| G1WholebodyBendPickMP-v0 | x  | v | x    
| G1WholebodyHandoverTeleop-v0  | x  | v  | v
| G1WholebodyLocomotionPickBetweenTablesTeleop-v0  | v | x  | v
| G1WholebodyTabletopGraspMP-v0  | x  | v  | x
| G1WholebodyXMoveBendPickTeleop-v0  | v  | x | v
| G1WholebodyXMovePickTeleop-v0  | v  | x  | v
| G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0  | v  | x  | v


## Download eval data

```
export task=G1WholebodyBendPickMP-v0
```

```
hf download USC-PSI-Lab/psi-data \
    simple-eval/$task.zip \
    --local-dir=data/evals \
    --repo-type=dataset

unzip data/evals/simple-eval/$task.zip -d data/evals/simple-eval
```

## Start Server

> Make sure you start the policy server first. If you run SIMPLE locally on a workstation, we suggest start VLAs models on a different PC for better performance.

> If the server is started on a remote server, run ssh port forward. eg., ssh -L 22086:localhost:22086 songlin@nebula100.

> Once port forward is done, open a new terminal to test if server is up curl -i http://localhost:22085/health

## Run client

```
MUJOCO_GL=egl uv run eval simple/FrankaTabletopGrasp-v0 \
    openvla \
    --host=127.0.0.1 \
    --port=21075 \
    --sim-mode=mujoco_isaac \
    --headless \
    --max-episode-steps=50
```

or use docker, you can optionally set gpu device usig `GPUs={device_id}`
```
GPUs=1 docker compose run eval simple/FrankaTabletopGrasp-v0 \
    openvla \
    --host=172.17.0.1 \
    --port=21075 \
    --sim-mode=mujoco_isaac \
    --headless \
    --max-episode-steps=50
```

Find results at `./data/evals/openvla`

## Decoupled-WBC tasks (e.g. totes shelf-to-table)

Whole-body teleop tasks recorded via `teleop_decoupled_wbc.py` (locomotion +
manipulation driven by the decoupled WBC pipeline, e.g.
`G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0`) are evaluated through
`eval-decoupled-wbc`, not the plain `eval` CLI above -- it bootstraps the
same Sonic/WBC lower-body policy and stabilizes the robot before handing
control to the manipulation policy. Baseline agents for this family live in
`src/simple/baselines/*_decoupled_wbc.py` (e.g. `gr00t_n16`, `psi0`,
`pi05`).

To compare policies fairly, run every policy against the exact same
`--data-dir` (a held-out set of recorded teleop episodes, not used in
training) and the same `--success-criteria`/`--num-episodes`:

```
uv run eval-decoupled-wbc simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0 \
    gr00t_n16 \
    train \
    --host=127.0.0.1 \
    --port=21075 \
    --data-format lerobot \
    --data-dir=data/datagen/simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0/level-0 \
    --eval-dir=data/evals_decoupled_wbc \
    --num-episodes=20 \
    --success-criteria=0.9 \
    --sim-mode=mujoco \
    --headless
```

Swap `gr00t_n16` for `psi0` or `pi05` (matching the `*_decoupled_wbc.py`
baseline module names) to evaluate the other target policies, keeping
every other flag identical. Find results at
`./data/evals_decoupled_wbc/<policy>`.
### Evaluating a policy trained on several datasets

A policy is often trained on more than one corpus -- e.g. the single-table
shelf-to-table recordings plus the later mirrored left/right ones. To evaluate
it you need starting states from *both*, which a single `--data-dir` cannot
express. Rather than concatenating them into a throwaway third dataset, list
them in a YAML and pass it with `--eval-config`:

```yaml
# configs/evals/shelf_to_dual_table.yaml
datasets:
  - name: single_table
    path: data/psi0_G1WholebodyLocomotionPickTotesShelfToTableTeleop
    prompt: task
  - name: mirror
    path: data/psi0_G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop
```

```
uv run eval-decoupled-wbc simple/G1PickUpToteFromShelfToDeskMirrorPsi0-v0 \
    psi0 \
    train \
    --eval-config configs/evals/shelf_to_dual_table.yaml \
    --num-episodes 20 \
    --seed 0 \
    --host=127.0.0.1 \
    --port=21075 \
    --success-criteria=0.9 \
    --headless
```

Each entry is a plain lerobot root -- a dir holding `meta/`, `data/`, `videos/`
-- which is what `postprocess_psi0_sonic.py` emits and what
`render-decoupled-wbc --save-dir` writes. A bare path string works too when the
defaults suffice.

- `--num-episodes` is the **total**, split evenly across datasets so a large
  corpus doesn't drown out a small one. A dataset with fewer episodes than its
  share hands the remainder back to the others.
- Episodes are drawn at random within each dataset from `--seed`, so two
  policies compared with the same config and seed see the same episodes.
- `--eval-config` implies `--data-format lerobot` and cannot be combined with
  `--data-dir`.
- Results are keyed `<source_name>__episode_<index>`, in `eval_stats.txt` and in
  the video filenames, so you can tell which dataset an episode came from.

**`prompt`** selects the language instruction for that dataset:

| Value | Behaviour |
| --- | --- |
| `recorded` (default) | replay the prompt stored with the episode |
| `task` | drop the episode's recorded language randomizer state so the task rebuilds the prompt from its own template during `reset()` |

Use `recorded` unless the stored prompt is under-specified for the environment
being evaluated. The single-table recordings are exactly that case: their prompt
is *"pick up the blue tote from the shelf and bring it to the table."*, which
names no side, while the mirrored environment scores strictly on delivering to
the **commanded** table. Replaying it would ask for something ambiguous and then
grade it against a randomly drawn side.

**`conditions`** overrides the per-episode values the task reads out of the
state_dict -- for the mirrored task, `pick_hand` (`left`/`right`/`both`) and
`target_side` (`left`/`right`):

```yaml
  - name: single_table
    path: data/psi0_G1WholebodyLocomotionPickTotesShelfToTableTeleop
    prompt: task
    conditions:
      pick_hand: right
      target_side: [left, right]
```

| Form | Behaviour |
| --- | --- |
| a scalar (`right`) | pins that value on every episode of the dataset |
| a list (`[left, right]`) | dealt round-robin across the dataset's episodes |
| `recorded` (default when the key is absent) | leaves the key out, so the episode's own value survives |

The list form is what makes an unbalanced corpus yield balanced *commands*: the
single-table captures carry no side at all, yet asking for either table is
equally fair, since only the starting scene is replayed -- not the trajectory.
Pinning `pick_hand` matters for a different reason: left unpinned, the task also
samples `both`, a bimanual grasp the policy may never have been taught, which
fails for reasons unrelated to following the language.

Overriding a condition requires `prompt: task`. Replaying a recorded prompt
while replacing the command it describes would tell the policy one thing and
score it on another, so that combination is rejected.

Note that `prompt: task` has to discard the recorded `dr_state_dict["language"]`
to work: the language randomizer's state is captured with everything else, and
`DRManager.load_state_dict` restores it verbatim, so an episode recorded under an
older task would otherwise keep imposing that task's wording no matter which
environment replays it.
