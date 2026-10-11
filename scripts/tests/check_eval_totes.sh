#!/usr/bin/env bash
set -euo pipefail

# Smoke test for the G1 totes shelf-to-table teleop task, mirroring
# check_eval.sh's plumbing but through eval-decoupled-wbc (this task is
# recorded/evaluated via the Sonic/WBC pipeline, not the plain `eval` CLI --
# see teleop_decoupled_wbc.py). Requires a recorded lerobot dataset produced
# by teleop_decoupled_wbc.py --record for this env id (there is no `_mp`
# variant/datagen path for this task, so check_datagen.sh cannot produce it).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${SIMPLE_TEST_OUTPUT_ROOT:-$ROOT_DIR/.test-output}"
ENV_ID="simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0"
SIM_MODE="${SIMPLE_SMOKE_SIM_MODE:-mujoco}"

# DATA_ROOT is the --save-dir passed to teleop_decoupled_wbc.py --record;
# that CLI writes episodes under "$save_dir/$ENV_ID/level-<dr_level>" (see
# run_save_dir in teleop_decoupled_wbc.py). replay_policy_server.py always
# reads from a hardcoded "level-0" subdir, so this smoke test only works
# against a level-0 recording.
DATA_ROOT="${SIMPLE_EVAL_TOTES_DATA_ROOT:-}"
if [[ -z "$DATA_ROOT" || ! -d "$DATA_ROOT" ]]; then
  echo "[check_eval_totes] set SIMPLE_EVAL_TOTES_DATA_ROOT to the --save-dir used with:" >&2
  echo "[check_eval_totes]   teleop_decoupled_wbc.py $ENV_ID --record --dr-level 0" >&2
  exit 1
fi
DATASET_DIR="$DATA_ROOT/$ENV_ID/level-0"
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[check_eval_totes] dataset dir not found: $DATASET_DIR" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)"
EVAL_ROOT="$OUT_ROOT/eval-totes-smoke-$RUN_ID"
SERVER_LOG="$EVAL_ROOT/replay-server.log"
PORT="${SIMPLE_EVAL_SMOKE_PORT:-21091}"

mkdir -p "$EVAL_ROOT"

echo "[check_eval_totes] dataset dir: $DATASET_DIR"
echo "[check_eval_totes] eval root: $EVAL_ROOT"
echo "[check_eval_totes] sim mode: $SIM_MODE"

if [[ -z "${ROBO_NIX_ACTIVE:-}" ]]; then
  echo "[check_eval_totes] run this script inside 'robo shell'" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
  uv venv --python "${UV_PYTHON:-python3.10}" "$UV_PROJECT_ENVIRONMENT"
fi

if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/eval-decoupled-wbc" ]]; then
  echo "[check_eval_totes] missing eval-decoupled-wbc entrypoint" >&2
  echo "[check_eval_totes] run: bash scripts/setup_python_env.sh" >&2
  exit 1
fi

"$UV_PROJECT_ENVIRONMENT/bin/python" ./scripts/tests/replay_policy_server.py \
  --data-root "$DATA_ROOT" \
  --env-id "$ENV_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
echo $! > "$EVAL_ROOT/server.pid"

for _ in $(seq 1 60); do
  if "$UV_PROJECT_ENVIRONMENT/bin/python" - <<PY
import json
import urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:$PORT/healthz", timeout=1) as resp:
        payload = json.load(resp)
    raise SystemExit(0 if payload.get('ok') else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi
  sleep 1
done

# Same --success-criteria as the task's own default (0.9), not the CLI's
# 0.7 default, so this smoke test exercises the actual threshold used for
# real policy evaluation (see the "fair comparison" plan notes).
"$UV_PROJECT_ENVIRONMENT/bin/eval-decoupled-wbc" "$ENV_ID" replay_policy train \
  --host 127.0.0.1 \
  --port "$PORT" \
  --data-format lerobot \
  --data-dir "$DATASET_DIR" \
  --eval-dir "$EVAL_ROOT" \
  --num-episodes 1 \
  --success-criteria 0.9 \
  --sim-mode "$SIM_MODE" \
  --headless

SERVER_PID="$(cat "$EVAL_ROOT/server.pid")"

ENV_ID_TAIL="${ENV_ID##*/}"
test -f "$EVAL_ROOT/eval_stats.txt"
test -d "$EVAL_ROOT/replay_policy/$ENV_ID_TAIL/train"
grep -q "success rate:" "$EVAL_ROOT/eval_stats.txt"
find "$EVAL_ROOT/replay_policy/$ENV_ID_TAIL/train" -type f \( -name '*.mp4' -o -name '*.png' \) | grep -q .

echo "[check_eval_totes] success"
echo "[check_eval_totes] eval root: $EVAL_ROOT"
