"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Shared helpers for the teleop-capture -> Hugging Face upload pipeline.
Not a console script itself; imported by `upload_teleop_session.py` and
`sync_and_render.py`.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_and_validate_session(session_dir: str | Path) -> dict:
    """Validate that an on-disk LeRobot session tree (as written by
    `teleop_decoupled_wbc.py` or `render_decoupled_wbc.py`) is upload-ready,
    and return its `metadata.json` contents.

    Raises ValueError with all problems found, rather than failing on the
    first one, so a caller can show an operator the full list at once.
    """
    session_dir = Path(session_dir)
    errors = []

    meta_json = session_dir / "metadata.json"
    info_json = session_dir / "meta" / "info.json"
    episodes_jsonl = session_dir / "meta" / "episodes.jsonl"

    for path in (meta_json, info_json, episodes_jsonl):
        if not path.exists():
            errors.append(f"missing {path}")
    if errors:
        raise ValueError(f"Session validation failed for {session_dir}: {errors}")

    with open(meta_json, "r") as f:
        metadata = json.load(f)
    with open(info_json, "r") as f:
        info = json.load(f)

    total_episodes = info.get("total_episodes", 0)
    if total_episodes < 1:
        errors.append(
            f"meta/info.json reports total_episodes={total_episodes} -- refusing to upload an empty session"
        )
    if metadata.get("num_episodes") in (None, 0):
        errors.append(
            f"metadata.json reports num_episodes={metadata.get('num_episodes')} -- "
            "session may not have finished cleanly (was it interrupted before any episode was saved?)"
        )

    parquet_files = list((session_dir / "data").rglob("*.parquet"))
    video_files = list((session_dir / "videos").rglob("*.mp4"))
    if len(parquet_files) != total_episodes:
        errors.append(f"parquet file count {len(parquet_files)} != total_episodes {total_episodes}")
    total_videos = info.get("total_videos", 0)
    if total_videos and len(video_files) != total_videos:
        errors.append(f"video file count {len(video_files)} != total_videos {total_videos}")

    for p in parquet_files + video_files:
        if p.stat().st_size == 0:
            errors.append(f"zero-byte file: {p}")

    if errors:
        raise ValueError(f"Session validation failed for {session_dir}: {errors}")

    return metadata
