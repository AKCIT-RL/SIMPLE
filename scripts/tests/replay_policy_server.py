#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any

import pandas as pd

from simple.baselines.client import ResponseMessage


def _resolve_dataset_root(data_root: Path, env_id: str) -> Path:
    direct = data_root / env_id
    if direct.exists():
        return direct

    stripped = env_id.split("/", 1)[1] if "/" in env_id else env_id
    stripped_path = data_root / stripped
    if stripped_path.exists():
        return stripped_path

    matches = sorted(path for path in data_root.glob(f"{stripped}*") if path.is_dir())
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not locate dataset for env_id={env_id!r} under {data_root}."
    )


def _load_episode_actions(dataset_root: Path, episode_index: int) -> list[list[float]]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    parquet_path = dataset_root / info["data_path"].format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    frame_table = pd.read_parquet(parquet_path)
    return [list(map(float, action)) for action in frame_table["action"].tolist()]


class ReplayStore:
    def __init__(self, actions: list[list[float]], chunk_size: int):
        if not actions:
            raise ValueError("Recorded dataset episode contains no actions.")
        self._actions = actions
        self._chunk_size = chunk_size
        self._session_steps: dict[str, int] = {}

    def act(self, history: dict[str, Any]) -> list[list[float]]:
        session_id = str(history.get("session_id", "default"))
        if history.get("reset"):
            self._session_steps[session_id] = 0

        start = self._session_steps.get(session_id, 0)
        if start >= len(self._actions):
            return []

        end = min(start + self._chunk_size, len(self._actions))
        self._session_steps[session_id] = end
        return self._actions[start:end]


class ReplayRequestHandler(BaseHTTPRequestHandler):
    store: ReplayStore
    request_log: Path | None = None

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/act":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        history = request.get("history") or {}
        if self.request_log is not None:
            # What the policy would actually be conditioned on. Lets a test
            # assert the prompt reaching the policy without needing a model.
            with open(self.request_log, "a") as log:
                log.write(json.dumps({
                    "reset": bool(history.get("reset")),
                    "instruction": request.get("instruction"),
                }) + "\n")
        actions = self.store.act(history)
        self._send_json(200, ResponseMessage(action=actions, err=0.0).serialize())

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded SIMPLE datagen actions over HTTP.")
    parser.add_argument("--data-root", help="Datagen run root, e.g. .test-output/datagen-smoke-... (resolved as <root>/<env_id>/level-0)")
    parser.add_argument("--dataset-root", help="Lerobot root to replay from, used as-is. Needed for roots without a level-0 layer, e.g. a psi0 dataset.")
    parser.add_argument("--env-id", default="simple/G1WholebodyBendPickMP-v0")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21090)
    parser.add_argument("--log-requests", help="Append one JSON line per /act request (reset flag + instruction) to this file.")
    args = parser.parse_args()

    if bool(args.data_root) == bool(args.dataset_root):
        parser.error("pass exactly one of --data-root or --dataset-root")
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = _resolve_dataset_root(Path(args.data_root), args.env_id) / "level-0"
    actions = _load_episode_actions(dataset_root, args.episode_index)
    store = ReplayStore(actions, chunk_size=args.chunk_size)

    class Handler(ReplayRequestHandler):
        pass

    Handler.store = store
    if args.log_requests:
        log_path = Path(args.log_requests)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        Handler.request_log = log_path
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[replay_policy_server] serving {len(actions)} recorded actions from "
        f"{dataset_root} on http://{args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
