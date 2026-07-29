# Run an Environment
Script Usage:
```
python scripts/test_env.py --help
```
Run with default parameters:
```
python scripts/test_env.py
```
By default, a video result will be recoded under `./output/test`.
You can open this folder to check the simulation results.

If you want to video be played using a `libx264` compatitable player. e.g., `VSCode`, please install `ffmpeg`
```
sudo apt-get install ffmpeg
```
> Tips: You might install `ffmpeg` to have video generated in `libx264` format, so that the video can be directly previewed in `VSCode`.

## Running without the Franka assets (G1 example)

The defaults above target the Franka tabletop task (`--env-id
simple/FrankaTabletopGrasp-v0`, `--robot-uid franka_fr3`), which needs the Franka
robot/assets. If those aren't set up, a bare `python scripts/test_env.py` fails
while building that robot — this is the most common first-run error.

To run one of the G1 industrial environments instead — they use the `g1_sonic`
robot and never touch the Franka assets — override the env, task, robot and
target object:

```bash
python scripts/test_env.py \
  --env-id simple/G1IndustrialScrewdriverToToteTeleop-v0 \
  --task g1_industrial_screwdriver_to_tote_teleop \
  --robot-uid g1_sonic \
  --target-object graspnet1b:19 \
  --sim-mode mujoco --no-webrtc
```

The four flags that steer away from Franka are `--env-id`, `--task`,
`--robot-uid g1_sonic` and `--target-object`. (`simple/G1IndustrialScrewToToteTeleop-v0`
with `--task g1_industrial_screw_to_tote_teleop` is an equivalent alternative.)

Notes:
- **Flags use dashes, not underscores** (Typer): `--env-id`, `--robot-uid`,
  `--target-object`, `--sim-mode`, `--no-headless`, `--no-webrtc`. A bare
  `--env_id` is not recognized.
- `--sim-mode mujoco` is the lightest option (no Isaac Sim boot); the default is
  `mujoco_isaac`, and `isaac` is the third choice.
- On a laptop with a discrete NVIDIA GPU you may need the render-offload
  env vars, e.g. `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
  CUDA_VISIBLE_DEVICES=0 python scripts/test_env.py ...`.
- List every registered env id / task uid with `python scripts/list_env.py`.

## Visualizing an industrial task in the Isaac warehouse

To *see* an industrial task rendered in Isaac Sim's warehouse — no teleoperation,
just the scene with the robot driven by random actions — run `test_env.py` with
`--sim-mode isaac --no-headless` (this opens the Isaac GUI):

```bash
python scripts/test_env.py \
  --env-id simple/G1IndustrialSortingTeleop-v0 \
  --task g1_industrial_sorting_teleop \
  --robot-uid g1_sonic \
  --target-object graspnet1b:27 \
  --sim-mode isaac --no-headless
```

Notes:
- The warehouse scene is **baked into these tasks** via their DR config
  (`scene_manager="warehouse"`, `room_choices=["warehouse:default"]`). The
  `--scene-uid` flag is *not* consumed by the industrial tasks, so there is no
  need to pass it — the task always loads `warehouse:default`.
- Swap `--env-id` / `--task` for any industrial task, e.g.
  `simple/G1IndustrialToteToRackTeleop-v0` / `g1_industrial_tote_to_rack_teleop`.
- `--sim-mode isaac` boots Isaac Sim (~1 min on first launch). Use
  `--sim-mode mujoco` instead for a quick MuJoCo-only preview without Isaac.

## Detailed Explanations

Common imports:

```
# Include this line to parse command line args
from simple.args import args

# Import all the built-in environments
import simple.envs 

# Import a wrapper for recording simulation videos
from simple.envs.wrappers import VideoRecorder
```

Create a [Gym-style](https://gymnasium.farama.org/) environment:
```
# Create a built-in environment
env = gym.make(
    "simple/FrankaTabletopGrasp-v0",
    task="franka_tabletop_grasp",
    robot_uid="franka_fr3",
    controller_uid="pd_joint_pos", 
    target_object="graspnet1b:63",
    sim_mode=args.sim_mode,
    max_episode_steps=args.max_episode_steps, 
    headless=args.headless,
)
```

There are a few import parameters here:

+ `task`=`simple/FrankaTabletopGrasp-v0`, pass the task uid here. All the built-in tasks can be listed by running
  
    ```
     python scripts/list_env.py
    ```

+ `robot_uid`=`franka_fr3`, pass the robot uid here. We currently support Aloha, Franka panda, research3, a finger extended panda ... [TODO]

+ `controller_uid`=`pd_joint_pos`, choose the controller method, currently supported `pd_joint_pos`, `pd_delta_eef` ... [TODO]

+ `max_episode_steps`=`args.max_episode_steps`. Maximum steps allowed for each episode.

+ `headless`=`[True|False]`. If set to false, `IsaacSim` 's GUI will show.

+ `sim_mode`=`args.sim_mode`. Available choices: `[mujoco|isaac|mujoco_isaac]`

+ `target_object`=`graspnet1b:63`, This is a task-specific parameter. In this case the target object's [asset uid]() to grasp. 


Main loop
```
observation, info = env.reset()

frames = []
episode_over = False
while not episode_over:
    # sample random actions 
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()
```