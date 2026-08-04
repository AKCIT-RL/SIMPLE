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