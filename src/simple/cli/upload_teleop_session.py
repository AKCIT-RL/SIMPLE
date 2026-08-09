"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Upload one raw teleop session (LeRobot-format, as written by
`teleop_decoupled_wbc.py --record`) to the raw-capture Hugging Face dataset repo.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

from simple.cli._hf_session_utils import load_and_validate_session

load_dotenv()


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def main(
    session_dir: Annotated[
        str, typer.Argument(help="Path to one session's LeRobot dataset root (contains meta/, data/, videos/, metadata.json).")
    ],
    repo_id: Annotated[str, typer.Option(envvar="HF_REPO_RAW")] = "",
    dry_run: Annotated[bool, typer.Option(help="Validate and print the target path, but don't upload.")] = False,
):
    """Validate then upload a raw teleop session directory to the raw HF dataset repo."""
    if not repo_id:
        raise typer.BadParameter(
            "No repo id given. Pass --repo-id or set HF_REPO_RAW in your .env."
        )

    session_path = Path(session_dir).resolve()
    metadata = load_and_validate_session(session_path)

    env_name = metadata["env_id"].split("/")[-1]
    target_path = (
        f"raw/{metadata['operator']}/{metadata['session_timestamp']}"
        f"__{env_name}__level-{metadata['dr_level']}"
    )

    print(f"[upload-teleop-session] {session_path} -> {repo_id}:{target_path}")
    if dry_run:
        print("[upload-teleop-session] --dry-run set, not uploading.")
        return

    token = _hf_token()
    if not token:
        raise typer.BadParameter(
            "No Hugging Face token found. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in your .env."
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(session_path),
        path_in_repo=target_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add raw session {target_path}",
    )

    metadata["status"] = "uploaded"
    with open(session_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[upload-teleop-session] Done. status=uploaded, local metadata.json updated.")


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
