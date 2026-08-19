"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

One-off recovery tool for sessions rendered via sync_and_render.py's
--episode-chunk-size path before the fix in src/simple/cli/render_decoupled_wbc.py:
each chunk ran render-decoupled-wbc as a fresh subprocess whose local
episodes_saved counter restarted at 0, so environment_config got rewritten
into the first chunk_size rows of meta/episodes.jsonl instead of the rows a
later chunk actually appended -- leaving those later episodes without an
environment_config, which postprocess_psi0_sonic.py then silently drops.

This recovers the missing environment_config from the corresponding session
in the *raw* HF repo (recorded by teleop-decoupled-wbc.py in a single
process, unaffected by the bug) and re-uploads only the patched
meta/episodes.jsonl -- no re-rendering needed. Only patches sessions where
the raw and rendered episode counts match 1:1 (output position == source
episode_index); sessions with a mismatch (e.g. a permanently-skipped crash
episode during chunked rendering) are reported but left untouched, since the
mapping can't be safely reconstructed after the fact.

Usage:
    python scripts/patch_missing_environment_config.py \
        --rendered-repo-id AKCITWMOPOC/G1WB_PickToteShelfToTable_LHRT \
        --raw-repo-id AKCITWMOPOC/G1WB_PickToteShelfToTable_LHRT-raw \
        [--dry-run]

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import argparse
import json
import os

from huggingface_hub import HfApi, hf_hub_download


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered-repo-id", required=True)
    parser.add_argument("--raw-repo-id", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't upload patches.")
    args = parser.parse_args()

    token = _hf_token()
    if not token:
        raise SystemExit("No HF_TOKEN/HUGGINGFACE_HUB_TOKEN set in the environment.")
    api = HfApi(token=token)

    files = api.list_repo_files(repo_id=args.rendered_repo_id, repo_type="dataset")
    session_prefixes = sorted({
        "/".join(f.split("/")[:3])
        for f in files
        if f.startswith("rendered/") and f.endswith("/metadata.json")
    })
    print(f"Found {len(session_prefixes)} rendered session(s) in {args.rendered_repo_id}.")

    for session_prefix in session_prefixes:
        print(f"\n=== {session_prefix} ===")
        meta_local = hf_hub_download(
            repo_id=args.rendered_repo_id, repo_type="dataset",
            filename=f"{session_prefix}/metadata.json", token=token,
        )
        with open(meta_local, "r") as f:
            rendered_metadata = json.load(f)
        raw_session_prefix = rendered_metadata["source_raw_path"]

        episodes_local = hf_hub_download(
            repo_id=args.rendered_repo_id, repo_type="dataset",
            filename=f"{session_prefix}/meta/episodes.jsonl", token=token,
        )
        rendered_episodes = _load_jsonl(episodes_local)

        missing = [i for i, e in enumerate(rendered_episodes) if not e.get("environment_config")]
        if not missing:
            print("  Nothing missing, skipping.")
            continue
        print(f"  Missing environment_config for {len(missing)}/{len(rendered_episodes)} episodes.")

        raw_episodes_local = hf_hub_download(
            repo_id=args.raw_repo_id, repo_type="dataset",
            filename=f"{raw_session_prefix}/meta/episodes.jsonl", token=token,
        )
        raw_episodes = _load_jsonl(raw_episodes_local)

        if len(raw_episodes) != len(rendered_episodes):
            print(
                f"  SKIPPING: raw session ({raw_session_prefix}) has {len(raw_episodes)} "
                f"episodes but rendered has {len(rendered_episodes)} -- counts don't match "
                "1:1, so output position can't be safely assumed to equal source "
                "episode_index (likely a permanently-skipped crash episode). Needs manual "
                "mapping."
            )
            continue

        patched = 0
        for i in missing:
            src_conf = raw_episodes[i].get("environment_config")
            if not src_conf:
                print(f"  WARNING: raw episode {i} also has no environment_config, cannot patch.")
                continue
            rendered_episodes[i]["environment_config"] = src_conf
            patched += 1
        print(f"  Patched {patched}/{len(missing)} missing entries.")

        if args.dry_run or patched == 0:
            continue

        _write_jsonl(episodes_local, rendered_episodes)
        api.upload_file(
            path_or_fileobj=episodes_local,
            path_in_repo=f"{session_prefix}/meta/episodes.jsonl",
            repo_id=args.rendered_repo_id,
            repo_type="dataset",
            commit_message=f"Patch missing environment_config for {session_prefix}",
        )
        print(f"  Uploaded patched episodes.jsonl -> {args.rendered_repo_id}:{session_prefix}/meta/episodes.jsonl")


if __name__ == "__main__":
    main()
