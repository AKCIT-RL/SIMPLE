# Teleop Capture → Hugging Face Pipeline

Architecture reference for the raw-capture → upload → re-render → rendered-upload pipeline.
For step-by-step operator instructions (recording + uploading), see
`docs/teleop_simple_study/vr_teleop_beginners_guide.md`, sections 4, 11, 12 — this doc is for
whoever maintains the pipeline or runs the render workstation.

## Overview

```
operator machine (MuJoCo only)          render workstation (MuJoCo + Isaac Sim)
─────────────────────────────           ──────────────────────────────────────
teleop-decoupled-wbc --record
        │
        ▼
  session dir (LeRobot/GR00T)
        │
        ▼
upload-teleop-session  ───────►  HF_REPO_RAW (raw/<operator>/<session>/)
                                          │
                                          ▼  (manual trigger)
                                  sync-and-render
                                          │  download + flatten
                                          ▼
                                  render-decoupled-wbc (subprocess)
                                          │
                                          ▼
                                  HF_REPO_RENDERED (rendered/<operator>/<session>/)
                                          │
                                          ▼
                                  render_status.json written back to HF_REPO_RAW
                                          │
                                          ▼  (same sync-and-render run, if anything was rendered)
                                  download *every* rendered session
                                          │
                                          ▼
                                  postprocess_psi0_sonic.py (subprocess, full rebuild)
                                          │
                                          ▼
                                  HF_REPO_PSI0 (mirrored, replaces prior build)
```

Three separate HF dataset repos (`HF_REPO_RAW`, `HF_REPO_RENDERED`, `HF_REPO_PSI0`), one shared
classic `HF_TOKEN` with write access to all three. Repo creation is a one-time manual admin
action — no script here auto-creates a repo (`create_repo` is intentionally not called anywhere
in this pipeline), so all three CLIs fail loudly if the target repo doesn't exist yet.

The capture already comes out in **LeRobot/GR00T format** at recording time
(`teleop_decoupled_wbc.py` writes via `Gr00tDataExporter`) — there is no separate "convert to
GR00T" stage. The only thing enforced between capture and upload is validation (below), not
format conversion.

## Session = one independent LeRobot dataset

`Gr00tDataExporter.create()` always builds a self-contained dataset tree per `save_root`
(`meta/`, `data/`, `videos/`) — it isn't designed to merge multiple independently-created roots
into one growing dataset. So each `teleop-decoupled-wbc --record` invocation creates its own
timestamped session directory:

```
data/teleop_decoupled_wbc/<env_id>/level-<dr_level>/sessions/<timestamp>__<operator>/
├── meta/{info.json,modality.json,episodes.jsonl,tasks.jsonl}
├── data/chunk-000/episode_*.parquet
├── videos/chunk-000/observation.images.ego_view/episode_*.mp4
└── metadata.json          # pipeline status tracking, see schema below
```

`HF_REPO_RAW`/`HF_REPO_RENDERED` are therefore **collections of independent per-session LeRobot
datasets**, not a single flat LeRobot dataset — loading `LeRobotDataset(repo_id=...)` on the repo
root won't work; target one session subdirectory. `HF_REPO_PSI0` is the exception: it's the one
place sessions get consolidated (see "psi0 training corpus" below).

## `metadata.json` schema

Written by `teleop_decoupled_wbc.py` at session start, finalized in its `finally` block
(covers both normal completion and `KeyboardInterrupt`), and updated in place by
`upload-teleop-session` / synthesized by `sync-and-render` for rendered output:

| field | set by | meaning |
| :--- | :--- | :--- |
| `schema_version` | capture | currently `1` |
| `operator` | capture | from `--operator` / `SIMPLE_OPERATOR` |
| `session_timestamp` | capture | `YYYYMMDD_HHMMSS`, session start |
| `env_id` | capture | full gym env id, e.g. `simple/G1Wholebody...-v0` |
| `dr_level` | capture | domain-randomization level |
| `task_prompt` | capture | natural-language instruction recorded with every frame; operator is prompted to confirm/override `task.instruction` (Enter keeps the default) right before recording starts |
| `status` | capture → upload → render | `raw_captured` → `uploaded` → (rendered output gets `rendered`) |
| `num_episodes` | capture (finalized on exit) | `None` until the run ends; validator rejects `None`/`0` |
| `created_at_utc` / `finished_at_utc` | capture | ISO8601 |
| `sim_mode`, `hostname` | capture | debugging context |
| `source_raw_path` | render | rendered-output only: `raw/...` prefix it came from |
| `rendered_at_utc` | render | rendered-output only |

`meta/info.json`'s `data_collection_info.teleoperator_username` (LeRobot-native, written by
`Gr00tDataExporter`) also carries the operator — `metadata.json` is SIMPLE-specific pipeline
state, kept separate rather than hand-editing the upstream schema.

## HF repo path layout

- Raw: `raw/<operator>/<timestamp>__<env_name>__level-<n>/` (env_name = `env_id` without the
  `simple/` namespace prefix)
- Rendered: `rendered/<operator>/<timestamp>__<env_name>__level-<n>/` — mirrors the raw path so
  a session can be found in both repos with the same suffix.
- Render-completion marker: `raw/<operator>/<timestamp>__<env_name>__level-<n>/render_status.json`
  — written back into the **raw** repo by `sync-and-render` after a successful rendered upload.
  This is the source of truth `sync-and-render` checks to avoid re-rendering.

## `upload-teleop-session`

`src/simple/cli/upload_teleop_session.py`. Validates the session (`_hf_session_utils.py`,
below) then `HfApi().upload_folder(...)` to `HF_REPO_RAW`. Updates local `metadata.json.status`
to `"uploaded"` on success. `--dry-run` to preview the target path without network calls.

## `sync-and-render`

`src/simple/cli/sync_and_render.py`. Pure orchestration (no `simple.envs`/Isaac import) — safe
to run in a plain process. Per run:

1. `list_repo_files` on `HF_REPO_RAW`, diff against `render_status.json` markers → pending
   sessions.
2. Download + flatten each pending session into `<local_stage_dir>/<env_id>/level-<n>` — the
   layout `render-decoupled-wbc --data-dir` expects.
3. Invoke `render-decoupled-wbc` **as a subprocess**, one per session — Isaac Sim's
   `SimulationApp` is a hard process-wide singleton
   (`src/simple/envs/base_dual_env.py`: `assert not _ISAAC_LOADED`), so this cannot be called
   in-process in a loop. Expect ~30-90s Isaac Sim cold start per session; batching isn't
   supported by `render-decoupled-wbc` today (one dataset in, one dataset out). Always passes
   `--isaac-background-usd` through to `render-decoupled-wbc` (`--isaac-background-usd`/
   `SIMPLE_ISAAC_BACKGROUND_USD`, defaults to the SimReady Warehouse USD) — without it, tasks
   that build their own scenario geometry render without the intended visual backdrop. See
   `docs/source/tutorials/isaac_warehouse_rendering.md` for how to confirm the right USD path
   for a given Isaac/Nucleus setup.
4. Synthesize `metadata.json` for the rendered tree (render-decoupled-wbc doesn't write one),
   validate it, upload to `HF_REPO_RENDERED`.
5. Write `render_status.json` back to the raw repo.
6. Delete local staged raw + rendered dirs (`--keep-local` to skip, for debugging).
7. If step 3 rendered at least one session this run (and `--skip-psi0` wasn't passed): rebuild
   the consolidated psi0 training corpus and upload it to `HF_REPO_PSI0` — see below. Skipped
   entirely if nothing new was rendered, since a rebuild is expensive and the corpus wouldn't
   have changed.

Idempotency: safe to re-run. Won't re-render a session once its `render_status.json` marker
exists in the raw repo. A crash between download and marker-write causes a redundant re-render
on the next run (acceptable for a manually-triggered, low-frequency operation) — render is a
pure function of the raw data, so redoing it is safe, just wasted GPU time.

Concurrency: raw uploads from different operators never collide (unique
operator+timestamp+env path per session).

## psi0 training corpus (`HF_REPO_PSI0`)

`scripts/postprocess_psi0_sonic.py` merges many independent LeRobot sessions into **one**
consolidated dataset in a different, training-ready schema ("psi0": `states`/`action` vectors
sliced into hand/arm/leg groups, `modality.json` in psi0's convention, video renamed to
`egocentric`). This is fundamentally different from every other step in this pipeline — everything
else produces one self-contained dataset per session; this is the one place many sessions get
flattened into a single trainable corpus. It only ever runs against **rendered** sessions (not
raw), since the point of the corpus is photorealistic training data.

`sync-and-render` runs it automatically, as a **full rebuild every time**, not an incremental
append:

1. `snapshot_download` the *entire* `HF_REPO_RENDERED` repo (not just this run's new sessions)
   into `--psi0-mirror-dir` (default `data/render_decoupled_wbc_mirror`). Each downloaded session
   lands at `<mirror>/rendered/<operator>/<session>/{meta,data,videos}` — exactly the shape
   `postprocess_psi0_sonic.py --sim-root` expects one glob level per operator/session:
   `<mirror>/rendered/*/*`.
2. Run `scripts/postprocess_psi0_sonic.py` as a subprocess (it's a standalone script, not a
   console entry point) with that glob as `--sim-root`, `--out-dir` = `--psi0-out-dir` (default
   `data/psi0_dataset`).
3. Upload `--psi0-out-dir` to `HF_REPO_PSI0` with `delete_patterns=["**"]` — a full mirror/replace,
   not an additive commit.

**Why full rebuild, not incremental**: `postprocess_psi0_sonic.py` assigns `episode_index`
sequentially by processing order across all matched `sim_root`s. Adding one new session and
appending its episodes on top of a previous partial build would risk misnumbered/duplicate
`episode_index` values, since the ordering isn't stable as new sessions are inserted. Rebuilding
from scratch every time and mirroring (not appending) to the repo is simpler and avoids that
whole class of bug, at the cost of re-downloading and re-processing the full corpus on every
`sync-and-render` run that rendered something new. This gets more expensive as the corpus grows —
acceptable for a manually-triggered, low-frequency operation today, but worth revisiting
(incremental append, or a separate `build-psi0-dataset` command decoupled from the render loop)
once corpus size makes a full rebuild noticeably slow.

**`--total_episodes` gotcha**: `postprocess_psi0_sonic.py` defaults to `--total_episodes=99` and
silently *stops processing* (not an error) once that many episodes are merged — a corpus that
grows past that cap would quietly lose sessions from the psi0 build without any error. `sync-and-
render` overrides this with `--psi0-total-episodes` defaulting to `100000`; raise it further if
your corpus ever approaches that.

Relevant `sync-and-render` options: `--psi0-repo-id`/`HF_REPO_PSI0`, `--psi0-mirror-dir`,
`--psi0-out-dir`, `--psi0-skip`, `--psi0-downsample`, `--psi0-fps`, `--psi0-total-episodes`,
`--psi0-video-key`, `--psi0-chunks-size`, `--skip-psi0` (opt out entirely, e.g. while iterating on
the render stage alone).

## Pre-upload validation (`src/simple/cli/_hf_session_utils.py`)

`load_and_validate_session(session_dir)` — shared by both CLIs, checks:
- `metadata.json`, `meta/info.json`, `meta/episodes.jsonl` all present.
- `total_episodes >= 1` (refuses empty sessions).
- `metadata.json.num_episodes` not `None`/`0`.
- parquet file count matches `total_episodes`; mp4 count matches `total_videos`.
- no zero-byte files.

Raises `ValueError` listing every problem found (not just the first), so an operator sees the
full picture at once.

## Admin setup checklist (one-time)

1. Create `HF_REPO_RAW`, `HF_REPO_RENDERED`, and `HF_REPO_PSI0` as HF dataset repos (visibility
   per your org's policy), e.g. `hf repo create USC-PSI-Lab/simple-teleop-raw --type dataset`.
2. Add a dataset card (`README.md`) to each repo — YAML frontmatter
   (`task_categories: [robotics]`, `tags: [LeRobot, teleoperation, G1]`). For raw/rendered, note
   that the repo is a collection of independent per-session LeRobot datasets, fps/features
   pointer to any session's `meta/info.json`. For psi0, note that it's a single consolidated,
   fully-rebuilt-on-each-sync corpus (not per-session) in the psi0 schema, and that its content
   can be entirely replaced by any `sync-and-render` run.
3. Distribute a single write-scoped `HF_TOKEN` to operators and to the render workstation via
   `.env` (never commit it — `.env` is gitignored).
4. Set `HF_REPO_RAW`/`HF_REPO_RENDERED`/`HF_REPO_PSI0` in `.env.sample` (already done) so new
   operators/workstations pick up the right repo ids by default.
