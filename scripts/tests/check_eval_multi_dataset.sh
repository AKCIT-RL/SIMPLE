#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for the --eval-config multi-dataset eval path.
#
# Runs the real simulator but needs no trained model: the policy is
# replay_policy_server.py, which replays one recorded episode's actions for
# every episode. Most episodes will be scored as failures -- that is expected
# and irrelevant here. What is under test is the plumbing: which episodes get
# run, from which dataset, under which prompt, and that parallel workers
# partition the plan instead of colliding.
#
# Sources default to the totes captures in this repo's data/ (each session dir
# is already a valid lerobot root). Override with the SIMPLE_EVAL_MD_SRC_*
# variables to point at rendered or psi0 datasets instead.
#
#   bash scripts/tests/check_eval_multi_dataset.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV="${SIMPLE_VENV:-${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv-merge}}"
OUT_ROOT="${SIMPLE_TEST_OUTPUT_ROOT:-$ROOT_DIR/.test-output}"
ENV_ID="${SIMPLE_EVAL_MD_ENV_ID:-simple/G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop-v0}"
SIM_MODE="${SIMPLE_SMOKE_SIM_MODE:-mujoco}"
PORT="${SIMPLE_EVAL_SMOKE_PORT:-21092}"
NUM_EPISODES="${SIMPLE_EVAL_MD_NUM_EPISODES:-6}"
MAX_STEPS="${SIMPLE_EVAL_MD_MAX_STEPS:-60}"
SEED=0

TELEOP_ROOT="data/teleop_decoupled_wbc/simple"
MIRROR_SESSIONS="$TELEOP_ROOT/G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop-v0/level-0/sessions"
SINGLE_SESSIONS="$TELEOP_ROOT/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0/level-0/sessions"

pick_session() {  # pick_session <sessions_dir> <nth>  -- nth session with >0 episodes
  local dir="$1" want="$2" seen=0
  [[ -d "$dir" ]] || return 1
  for session in "$dir"/*/; do
    [[ -f "$session/meta/info.json" ]] || continue
    local n
    n="$("$VENV/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))['total_episodes'])" "$session/meta/info.json")"
    [[ "$n" -gt 0 ]] || continue
    seen=$((seen + 1))
    if [[ "$seen" -eq "$want" ]]; then
      echo "${session%/}"
      return 0
    fi
  done
  return 1
}

SRC_A="${SIMPLE_EVAL_MD_SRC_A:-$(pick_session "$MIRROR_SESSIONS" 1 || true)}"
SRC_B="${SIMPLE_EVAL_MD_SRC_B:-$(pick_session "$MIRROR_SESSIONS" 2 || true)}"
SRC_C="${SIMPLE_EVAL_MD_SRC_C:-$(pick_session "$SINGLE_SESSIONS" 1 || true)}"

if [[ -z "$SRC_A" || -z "$SRC_B" ]]; then
  echo "[check_eval_md] need at least two non-empty capture sessions under $MIRROR_SESSIONS" >&2
  echo "[check_eval_md] override with SIMPLE_EVAL_MD_SRC_A / SIMPLE_EVAL_MD_SRC_B" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/eval-decoupled-wbc" ]]; then
  echo "[check_eval_md] missing $VENV/bin/eval-decoupled-wbc" >&2
  echo "[check_eval_md] run: bash scripts/setup_python_env.sh (or set SIMPLE_VENV)" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)"
EVAL_ROOT="$OUT_ROOT/eval-multi-dataset-$RUN_ID"
mkdir -p "$EVAL_ROOT"
CONFIG="$EVAL_ROOT/eval_sources.yaml"
SERVER_LOG="$EVAL_ROOT/replay-server.log"
REQUEST_LOG="$EVAL_ROOT/policy-requests.jsonl"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

{
  echo "datasets:"
  echo "  - name: mirror_a"
  echo "    path: $SRC_A"
  echo "  - name: mirror_b"
  echo "    path: $SRC_B"
  if [[ -n "$SRC_C" ]]; then
    # Single-table capture: its recorded prompt names no side, so let the task
    # rebuild a side-aware one. Also the "dataset smaller than its quota" case.
    echo "  - name: single_table"
    echo "    path: $SRC_C"
    echo "    prompt: task"
    echo "    conditions:"
    echo "      pick_hand: right"
    echo "      target_side: [left, right]"
  fi
} > "$CONFIG"

echo "[check_eval_md] env id:    $ENV_ID"
echo "[check_eval_md] sim mode:  $SIM_MODE"
echo "[check_eval_md] eval root: $EVAL_ROOT"
echo "[check_eval_md] config:"
sed 's/^/    /' "$CONFIG"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$VENV/bin/python" ./scripts/tests/replay_policy_server.py \
  --dataset-root "$SRC_A" \
  --env-id "$ENV_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --log-requests "$REQUEST_LOG" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do
  if "$VENV/bin/python" - <<PY
import json, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:$PORT/healthz", timeout=1) as resp:
        raise SystemExit(0 if json.load(resp).get("ok") else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi
  sleep 1
done

run_eval() {  # run_eval <subdir> <extra args...>
  local subdir="$1"; shift
  "$VENV/bin/eval-decoupled-wbc" "$ENV_ID" replay_policy train \
    --host 127.0.0.1 \
    --port "$PORT" \
    --eval-config "$CONFIG" \
    --num-episodes "$NUM_EPISODES" \
    --seed "$SEED" \
    --max-episode-steps "$MAX_STEPS" \
    --success-criteria 0.9 \
    --eval-dir "$EVAL_ROOT/$subdir" \
    --sim-mode "$SIM_MODE" \
    --headless \
    "$@"
}

echo ""
echo "[check_eval_md] === run 1: single worker ==="
run_eval single

echo ""
echo "[check_eval_md] === run 2: two workers, same seed ==="
run_eval parallel --num-workers 2

echo ""
echo "[check_eval_md] === run 3: legacy --data-dir path (no regression) ==="
"$VENV/bin/eval-decoupled-wbc" "$ENV_ID" replay_policy train \
  --host 127.0.0.1 \
  --port "$PORT" \
  --data-format lerobot \
  --data-dir "$SRC_A" \
  --num-episodes 1 \
  --max-episode-steps "$MAX_STEPS" \
  --success-criteria 0.9 \
  --eval-dir "$EVAL_ROOT/legacy" \
  --sim-mode "$SIM_MODE" \
  --headless

echo ""
echo "[check_eval_md] === assertions ==="
ENV_ID_TAIL="${ENV_ID##*/}"
EVAL_ROOT="$EVAL_ROOT" ENV_ID_TAIL="$ENV_ID_TAIL" NUM_EPISODES="$NUM_EPISODES" \
  "$VENV/bin/python" ./scripts/tests/assert_eval_multi_dataset.py

echo "[check_eval_md] success"
echo "[check_eval_md] eval root: $EVAL_ROOT"
