"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Manually-triggered orchestration script for the render-workstation side of the
teleop-capture -> Hugging Face pipeline:

    1. List raw sessions in the raw HF repo that are fully uploaded
       (metadata.json status == "uploaded") and not yet rendered
       (no raw/<session>/render_status.json marker).
    2. Download each pending session and stage it in the layout
       `render-decoupled-wbc` expects (a dir containing meta/, data/, videos/).
    3. Invoke `render-decoupled-wbc` as a subprocess (Isaac Sim's SimulationApp
       is a process-wide singleton, so this cannot be called in-process in a
       loop -- see src/simple/envs/base_dual_env.py).
    4. Synthesize a metadata.json for the rendered output, validate it, and
       upload it to the rendered HF repo.
    5. Write a render_status.json marker back into the raw repo so a re-run of
       this script doesn't re-render the same session.
    6. If anything was rendered this run: download *every* session in the
       rendered HF repo (not just this run's), rebuild the consolidated
       "psi0" training corpus (scripts/postprocess_psi0_sonic.py, which
       merges many independent LeRobot sessions into one dataset -- Gr00t's
       per-session exporter can't do this itself), and mirror the result to a
       third HF repo. This is a full rebuild every time, not an incremental
       append -- simplest way to stay correct given postprocess_psi0_sonic.py
       reassigns episode_index by processing order, so partial/incremental
       uploads would risk stale, wrongly-numbered files in the repo.

This script does not import simple.envs or touch Isaac Sim itself -- it is
pure orchestration and can run in a plain Python process.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

from simple.cli._hf_session_utils import load_and_validate_session

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[3]


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _list_pending_sessions(api, raw_repo_id: str) -> list[str]:
    """Return `raw/<operator>/<session>` prefixes that are status=uploaded and
    have no render_status.json marker yet."""
    files = api.list_repo_files(repo_id=raw_repo_id, repo_type="dataset")
    metadata_files = [f for f in files if f.startswith("raw/") and f.endswith("/metadata.json")]
    marker_prefixes = {f[: -len("/render_status.json")] for f in files if f.endswith("/render_status.json")}

    pending = []
    for meta_path in metadata_files:
        session_prefix = meta_path[: -len("/metadata.json")]
        if session_prefix in marker_prefixes:
            continue
        pending.append(session_prefix)
    return pending


def _stage_session(api, raw_repo_id: str, session_prefix: str, local_stage_dir: Path) -> tuple[Path, dict]:
    """Download one raw session and flatten it into <local_stage_dir>/<env_id>/level-<n>,
    the layout render-decoupled-wbc's --data-dir expects. Returns (staged_dir, metadata)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    meta_local = hf_hub_download(
        repo_id=raw_repo_id, repo_type="dataset", filename=f"{session_prefix}/metadata.json",
        token=_hf_token(),
    )
    with open(meta_local, "r") as f:
        metadata = json.load(f)

    download_root = local_stage_dir / "_download" / session_prefix
    snapshot_download(
        repo_id=raw_repo_id,
        repo_type="dataset",
        allow_patterns=[f"{session_prefix}/**"],
        local_dir=str(local_stage_dir / "_download"),
        token=_hf_token(),
    )

    staged_dir = local_stage_dir / metadata["env_id"] / f"level-{metadata['dr_level']}"
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)
    for name in ("meta", "data", "videos"):
        shutil.move(str(download_root / name), str(staged_dir / name))
    shutil.rmtree(local_stage_dir / "_download", ignore_errors=True)

    return staged_dir, metadata


def _run_render(env_id: str, staged_dir: Path, render_save_root: Path, sim_mode: str, headless: bool, dr_level: int):
    cmd = [
        "render-decoupled-wbc", env_id,
        "--data-dir", str(staged_dir),
        "--sim-mode", sim_mode,
        "--headless" if headless else "--no-headless",
        "--record",
        "--save-dir", str(render_save_root),
        "--dr-level", str(dr_level),
    ]
    print(f"[sync-and-render] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return render_save_root / env_id / f"level-{dr_level}"


def _download_all_rendered_sessions(rendered_repo_id: str, mirror_dir: Path) -> Path:
    """Mirror the full rendered HF repo locally. Returns the dir containing one
    subdir per session (each holding meta/, data/, videos/ directly) -- exactly
    the shape postprocess_psi0_sonic.py's --sim-root glob expects."""
    from huggingface_hub import snapshot_download

    if mirror_dir.exists():
        shutil.rmtree(mirror_dir)
    snapshot_download(
        repo_id=rendered_repo_id,
        repo_type="dataset",
        allow_patterns=["rendered/**"],
        local_dir=str(mirror_dir),
        token=_hf_token(),
    )
    return mirror_dir / "rendered"


def _rebuild_psi0_dataset(
    rendered_sessions_root: Path,
    psi0_out_dir: Path,
    skip: int,
    downsample: int,
    fps: int,
    total_episodes: int,
    video_key: str,
    chunks_size: int,
):
    """Merge every downloaded rendered session into one consolidated psi0-format
    dataset via scripts/postprocess_psi0_sonic.py. Always a full rebuild from
    scratch (see module docstring) -- psi0_out_dir is wiped first."""
    if psi0_out_dir.exists():
        shutil.rmtree(psi0_out_dir)
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "postprocess_psi0_sonic.py"),
        "--sim-root", str(rendered_sessions_root / "*" / "*"),
        "--out-dir", str(psi0_out_dir),
        "--skip", str(skip),
        "--downsample", str(downsample),
        "--fps", str(fps),
        "--total_episodes", str(total_episodes),
        "--video-key", video_key,
        "--chunks-size", str(chunks_size),
    ]
    print(f"[sync-and-render] rebuilding psi0 dataset: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _upload_psi0_dataset(api, psi0_repo_id: str, psi0_out_dir: Path):
    print(f"[sync-and-render] uploading consolidated psi0 dataset -> {psi0_repo_id}")
    # delete_patterns mirrors the repo to exactly this build's contents --
    # episode_index assignment in postprocess_psi0_sonic.py depends on
    # processing order, so a plain additive upload could leave stale,
    # wrongly-numbered files from a previous build sitting in the repo.
    api.upload_folder(
        folder_path=str(psi0_out_dir),
        repo_id=psi0_repo_id,
        repo_type="dataset",
        commit_message="Rebuild consolidated psi0 dataset",
        delete_patterns=["**"],
    )


def main(
    raw_repo_id: Annotated[str, typer.Option(envvar="HF_REPO_RAW")] = "",
    rendered_repo_id: Annotated[str, typer.Option(envvar="HF_REPO_RENDERED")] = "",
    psi0_repo_id: Annotated[str, typer.Option(envvar="HF_REPO_PSI0", help="Consolidated training-corpus repo. Rebuilt from every rendered session after this run, unless --skip-psi0.")] = "",
    local_stage_dir: Annotated[str, typer.Option()] = "data/teleop_decoupled_wbc",
    render_save_dir: Annotated[str, typer.Option()] = "data/render_decoupled_wbc",
    psi0_mirror_dir: Annotated[str, typer.Option(help="Local mirror of the full rendered repo, used as postprocess_psi0_sonic.py's input.")] = "data/render_decoupled_wbc_mirror",
    psi0_out_dir: Annotated[str, typer.Option()] = "data/psi0_dataset",
    psi0_skip: Annotated[int, typer.Option(help="Frames to skip at the start of each episode, passed to postprocess_psi0_sonic.py.")] = 0,
    psi0_downsample: Annotated[int, typer.Option()] = 1,
    psi0_fps: Annotated[int, typer.Option()] = 50,
    psi0_total_episodes: Annotated[int, typer.Option(help="Cap on merged episodes. postprocess_psi0_sonic.py silently stops past this -- keep it well above your corpus size.")] = 100_000,
    psi0_video_key: Annotated[str, typer.Option()] = "observation.images.ego_view",
    psi0_chunks_size: Annotated[int, typer.Option()] = 1000,
    skip_psi0: Annotated[bool, typer.Option(help="Skip the psi0 rebuild/upload stage entirely.")] = False,
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = True,
    limit: Annotated[int, typer.Option(help="Max sessions to process this run, -1 = all pending.")] = -1,
    keep_local: Annotated[bool, typer.Option(help="Don't delete staged raw/rendered dirs after upload.")] = False,
    dry_run: Annotated[bool, typer.Option(help="List pending sessions and exit, no download/render/upload.")] = False,
):
    """Sync pending raw sessions from the raw HF repo, re-render them with Isaac
    Sim, and upload the result to the rendered HF repo. Safe to re-run: already
    rendered sessions are skipped via render_status.json markers in the raw repo."""
    if not raw_repo_id or not rendered_repo_id:
        raise typer.BadParameter(
            "Both --raw-repo-id/HF_REPO_RAW and --rendered-repo-id/HF_REPO_RENDERED are required."
        )
    if not skip_psi0 and not psi0_repo_id:
        raise typer.BadParameter(
            "No psi0 repo id given. Pass --psi0-repo-id, set HF_REPO_PSI0 in your .env, or pass --skip-psi0."
        )
    token = _hf_token()
    if not token:
        raise typer.BadParameter(
            "No Hugging Face token found. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in your .env."
        )

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    pending = _list_pending_sessions(api, raw_repo_id)
    print(f"[sync-and-render] {len(pending)} session(s) pending render.")
    for session_prefix in pending:
        print(f"  - {session_prefix}")

    if dry_run:
        return

    if limit >= 0:
        pending = pending[:limit]

    local_stage_dir = Path(local_stage_dir).resolve()
    render_save_dir = Path(render_save_dir).resolve()

    for session_prefix in pending:
        print(f"\n[sync-and-render] Processing {session_prefix} ...")
        staged_dir, raw_metadata = _stage_session(api, raw_repo_id, session_prefix, local_stage_dir)

        # Unique --save-dir per session so concurrent env_id/level combos across
        # sessions don't append into the same Gr00tDataExporter root.
        render_save_root = render_save_dir / raw_metadata["operator"] / raw_metadata["session_timestamp"]
        rendered_dir = _run_render(
            raw_metadata["env_id"], staged_dir, render_save_root, sim_mode, headless, raw_metadata["dr_level"],
        )

        info_path = rendered_dir / "meta" / "info.json"
        with open(info_path, "r") as f:
            rendered_info = json.load(f)

        rendered_metadata = {
            "schema_version": 1,
            "source_raw_path": session_prefix,
            "operator": raw_metadata["operator"],
            "session_timestamp": raw_metadata["session_timestamp"],
            "env_id": raw_metadata["env_id"],
            "dr_level": raw_metadata["dr_level"],
            "status": "rendered",
            "num_episodes": rendered_info.get("total_episodes"),
            "rendered_at_utc": datetime.utcnow().isoformat() + "Z",
        }
        with open(rendered_dir / "metadata.json", "w") as f:
            json.dump(rendered_metadata, f, indent=2)

        load_and_validate_session(rendered_dir)

        env_name = raw_metadata["env_id"].split("/")[-1]
        rendered_target_path = (
            f"rendered/{raw_metadata['operator']}/{raw_metadata['session_timestamp']}"
            f"__{env_name}__level-{raw_metadata['dr_level']}"
        )
        print(f"[sync-and-render] uploading rendered session -> {rendered_repo_id}:{rendered_target_path}")
        api.upload_folder(
            folder_path=str(rendered_dir),
            path_in_repo=rendered_target_path,
            repo_id=rendered_repo_id,
            repo_type="dataset",
            commit_message=f"Add rendered session {rendered_target_path}",
        )

        # Mark the source raw session as rendered so the next run skips it.
        render_status = {
            "status": "rendered",
            "rendered_at_utc": rendered_metadata["rendered_at_utc"],
            "rendered_repo_id": rendered_repo_id,
            "rendered_path": rendered_target_path,
        }
        status_local_path = rendered_dir / "render_status.json"
        with open(status_local_path, "w") as f:
            json.dump(render_status, f, indent=2)
        api.upload_file(
            path_or_fileobj=str(status_local_path),
            path_in_repo=f"{session_prefix}/render_status.json",
            repo_id=raw_repo_id,
            repo_type="dataset",
            commit_message=f"Mark {session_prefix} as rendered",
        )

        if not keep_local:
            shutil.rmtree(staged_dir, ignore_errors=True)
            shutil.rmtree(rendered_dir, ignore_errors=True)

        print(f"[sync-and-render] Done with {session_prefix}.")

    if pending and not skip_psi0:
        print(f"\n[sync-and-render] Rebuilding consolidated psi0 dataset from all of {rendered_repo_id} ...")
        rendered_sessions_root = _download_all_rendered_sessions(rendered_repo_id, Path(psi0_mirror_dir).resolve())
        _rebuild_psi0_dataset(
            rendered_sessions_root,
            Path(psi0_out_dir).resolve(),
            skip=psi0_skip,
            downsample=psi0_downsample,
            fps=psi0_fps,
            total_episodes=psi0_total_episodes,
            video_key=psi0_video_key,
            chunks_size=psi0_chunks_size,
        )
        _upload_psi0_dataset(api, psi0_repo_id, Path(psi0_out_dir).resolve())
        if not keep_local:
            shutil.rmtree(Path(psi0_mirror_dir).resolve(), ignore_errors=True)
            shutil.rmtree(Path(psi0_out_dir).resolve(), ignore_errors=True)
        print("[sync-and-render] psi0 dataset rebuilt and uploaded.")


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
