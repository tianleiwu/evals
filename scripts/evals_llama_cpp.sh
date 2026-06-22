#!/bin/bash
# =============================================================================
# evals_llama_cpp.sh
#
# Evaluate MMLU accuracy for gpt-oss-20b served by llama.cpp.
#
# Pipeline:
#   1. Launch llama.cpp's `llama-server` with the gpt-oss-20b GGUF model. The
#      server exposes an OpenAI-compatible /v1/chat/completions endpoint.
#   2. Wait for the server /health endpoint to report ready.
#   3. Run the OpenAI `evals` harness (match_mmlu) against that endpoint via a
#      generated registry completion-fn that points at the local server and
#      extracts the A/B/C/D letter from the (possibly reasoning) response.
#   4. Parse + print the final accuracy and write a small summary report.
#
# The server is always shut down on exit (success, failure, or Ctrl-C).
#
# Usage:
#   ./evals_llama_cpp.sh                       # full MMLU (all subjects)
#   ./evals_llama_cpp.sh --max-samples 200     # cap total samples
#   ./evals_llama_cpp.sh --eval match_mmlu_anatomy
#   ./evals_llama_cpp.sh --reasoning-effort low
#   ./evals_llama_cpp.sh --no-server           # reuse an already-running server
#
# Options:
#   --eval NAME            Eval to run (default: match_mmlu = full MMLU).
#   --max-samples N        Cap number of samples (default: all).
#   --port N               llama-server port (default: 8080).
#   --ngl N                GPU layers to offload (default: 99 = all).
#   --ctx N                Total context size (default: 16384).
#   --parallel N           llama-server slots / concurrent requests (default: 4).
#   --reasoning-effort E   gpt-oss reasoning effort: low|medium|high (default: model default).
#   --no-server            Do not start/stop llama-server; use an existing one.
#   --outdir DIR           Output dir (default: timestamped under ~/eval_runs).
#   -h | --help            Show this help.
#
# Most CONFIG values below can be overridden via environment variables, e.g.:
#   MODEL_GGUF=/path/to/model.gguf VENV=/path/to/venv ./evals_llama_cpp.sh
# =============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# CONFIG (override via environment)
# ----------------------------------------------------------------------------
EVALS_DIR="${EVALS_DIR:-~/evals}"
VENV="${VENV:-~/venv}"

LLAMA_DIR="${LLAMA_DIR:-~/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
MODEL_GGUF="${MODEL_GGUF:-~/gguf/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf}"
MODEL_NAME="${MODEL_NAME:-gpt-oss-20b}"

# CUDA libs for the CUDA-enabled llama-server build.
CUDA_HOME="${CUDA_HOME:-~/cuda13.0}"
CUDNN_HOME="${CUDNN_HOME:-~/cudnn_9.19_cuda13}"

# Server runtime params
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
NGL="${NGL:-99}"
CTX="${CTX:-16384}"
PARALLEL="${PARALLEL:-4}"
FLASH_ATTN="${FLASH_ATTN:-auto}"      # on|off|auto
REASONING_EFFORT="${REASONING_EFFORT:-}"   # ""|low|medium|high
MAX_TOKENS="${MAX_TOKENS:-3072}"

# Eval params
EVAL="${EVAL:-match_mmlu}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
START_SERVER="${START_SERVER:-1}"

# Output
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-~/eval_runs/llamacpp_$TS}"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log()  { printf '\033[1;34m[evals-llamacpp]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[evals-llamacpp][warn]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[evals-llamacpp][error]\033[0m %s\n' "$*" >&2; exit 1; }

PY() { "$VENV/bin/python" "$@"; }

usage() {
    awk 'NR>1 && /^# / {sub(/^# ?/,""); print} /^set -euo/ {exit}' "$0"
    exit 0
}

# ----------------------------------------------------------------------------
# Arg parsing
# ----------------------------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --eval)             EVAL="$2"; shift 2 ;;
        --max-samples)      MAX_SAMPLES="$2"; shift 2 ;;
        --port)             PORT="$2"; shift 2 ;;
        --ngl)              NGL="$2"; shift 2 ;;
        --ctx)              CTX="$2"; shift 2 ;;
        --parallel)         PARALLEL="$2"; shift 2 ;;
        --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
        --no-server)        START_SERVER=0; shift ;;
        --outdir)           OUTDIR="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *)                  fail "Unknown argument: $1 (use --help)" ;;
    esac
done

API_BASE="http://$HOST:$PORT/v1"

# ----------------------------------------------------------------------------
# Pre-flight checks
# ----------------------------------------------------------------------------
[[ -x "$VENV/bin/oaieval" ]] || fail "oaieval not found in venv: $VENV (need evals installed)"
OPENAI_API_KEY=dummy PY -c "import evals, openai" >/dev/null 2>&1 \
    || fail "venv $VENV cannot import evals+openai"
if [[ "$START_SERVER" == "1" ]]; then
    [[ -x "$LLAMA_SERVER" ]] || fail "llama-server not found/executable: $LLAMA_SERVER"
    [[ -f "$MODEL_GGUF"   ]] || fail "GGUF model not found: $MODEL_GGUF"
fi

mkdir -p "$OUTDIR"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"   # evals registry builds OpenAI() at import
export EVALS_THREADS="${EVALS_THREADS:-$PARALLEL}" # match server slots for throughput
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"

# ----------------------------------------------------------------------------
# llama-server lifecycle
# ----------------------------------------------------------------------------
SERVER_PID=""
SERVER_LOG="$OUTDIR/llama_server.log"

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping llama-server (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
}
trap stop_server EXIT INT TERM

start_server() {
    log "Launching llama-server"
    log "  model : $MODEL_GGUF"
    log "  api   : $API_BASE  (ngl=$NGL ctx=$CTX slots=$PARALLEL fa=$FLASH_ATTN)"

    local extra=()
    [[ -n "$REASONING_EFFORT" ]] && \
        extra+=(--chat-template-kwargs "{\"reasoning_effort\":\"$REASONING_EFFORT\"}")

    "$LLAMA_SERVER" \
        -m "$MODEL_GGUF" \
        --alias "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        -ngl "$NGL" \
        -c "$CTX" \
        -np "$PARALLEL" \
        -fa "$FLASH_ATTN" \
        --jinja \
        "${extra[@]}" \
        >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    log "llama-server pid $SERVER_PID (log: $SERVER_LOG)"
}

wait_health() {
    log "Waiting for server health at $API_BASE ..."
    local url="http://$HOST:$PORT/health"
    for i in $(seq 1 240); do
        if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            warn "llama-server exited early; last log lines:"
            tail -n 30 "$SERVER_LOG" >&2 || true
            fail "llama-server failed to start"
        fi
        if curl -fsS "$url" 2>/dev/null | grep -q '"status"\s*:\s*"ok"'; then
            log "Server is ready (after ~${i}s)"
            return 0
        fi
        sleep 1
    done
    tail -n 30 "$SERVER_LOG" >&2 || true
    fail "Server did not become healthy in time"
}

if [[ "$START_SERVER" == "1" ]]; then
    start_server
    wait_health
else
    log "Using existing server at $API_BASE (--no-server)"
fi

# ----------------------------------------------------------------------------
# Generate a temporary registry completion-fn pointing at llama-server.
# Uses the choice-letter fn so MMLU's Match grader sees a clean A/B/C/D answer
# even when gpt-oss emits reasoning content.
# ----------------------------------------------------------------------------
REG_DIR="$OUTDIR/_registry"
mkdir -p "$REG_DIR/completion_fns"
COMPLETION_FN="llamacpp/$MODEL_NAME"
cat > "$REG_DIR/completion_fns/llama_cpp.yaml" <<YAML
$COMPLETION_FN:
  class: evals.completion_fns.openai_chat_choice_letter:OpenAIChatChoiceLetterFn
  args:
    model: $MODEL_NAME
    api_base: $API_BASE
    api_key: dummy
    extra_options:
      temperature: 0
      max_tokens: $MAX_TOKENS
YAML
log "Registry completion-fn: $COMPLETION_FN -> $API_BASE"

# ----------------------------------------------------------------------------
# Run the eval
# ----------------------------------------------------------------------------
RECORD="$OUTDIR/${EVAL}.jsonl"
EVAL_LOG="$OUTDIR/${EVAL}.log"

MS_ARG=()
[[ -n "$MAX_SAMPLES" ]] && MS_ARG=(--max_samples "$MAX_SAMPLES")

log "Running eval '$EVAL' (max_samples=${MAX_SAMPLES:-all}, threads=$EVALS_THREADS)"
set +e
( cd "$EVALS_DIR" && "$VENV/bin/oaieval" "$COMPLETION_FN" "$EVAL" \
    --registry_path "$REG_DIR" \
    --record_path "$RECORD" \
    "${MS_ARG[@]}" ) 2>&1 | tee "$EVAL_LOG"
EVAL_RC=${PIPESTATUS[0]}
set -e
[[ "$EVAL_RC" -eq 0 ]] || warn "oaieval exited with code $EVAL_RC (see $EVAL_LOG)"

# ----------------------------------------------------------------------------
# Parse accuracy from the record's final_report and write a summary.
# ----------------------------------------------------------------------------
ACC="$(RECORD="$RECORD" PY - <<'PYEOF'
import json, os
rec = os.environ["RECORD"]
acc = None
try:
    with open(rec) as f:
        for line in f:
            line = line.strip()
            if not line or '"final_report"' not in line:
                continue
            obj = json.loads(line)
            fr = obj.get("final_report") or {}
            if "accuracy" in fr:
                acc = fr["accuracy"]
except FileNotFoundError:
    pass
print(acc if acc is not None else "")
PYEOF
)"
[[ -n "$ACC" ]] || ACC="$(grep -oE '"accuracy"\s*:\s*[0-9.]+' "$EVAL_LOG" 2>/dev/null | grep -oE '[0-9.]+' | tail -1)"
ACC="${ACC:-CHECK_LOG}"

REPORT="$OUTDIR/summary_report.md"
{
    echo "# MMLU (llama.cpp) — $MODEL_NAME"
    echo
    echo "- Date: $(date)"
    echo "- Model GGUF: \`$MODEL_GGUF\`"
    echo "- Server: $API_BASE (ngl=$NGL, ctx=$CTX, slots=$PARALLEL, fa=$FLASH_ATTN)"
    [[ -n "$REASONING_EFFORT" ]] && echo "- Reasoning effort: $REASONING_EFFORT"
    echo "- Eval: \`$EVAL\`  (max_samples=${MAX_SAMPLES:-all})"
    echo "- Completion fn: \`$COMPLETION_FN\` (choice-letter extractor)"
    echo
    echo "## Result"
    echo
    echo "| eval | accuracy |"
    echo "|------|----------|"
    echo "| $EVAL | $ACC |"
    echo
    echo "Record: \`$RECORD\`"
    echo "Log:    \`$EVAL_LOG\`"
} > "$REPORT"

echo
log "==================== RESULT ===================="
log "  $EVAL accuracy: $ACC"
log "  report: $REPORT"
log "================================================"

[[ "$ACC" != "CHECK_LOG" && "$EVAL_RC" -eq 0 ]] || exit 1
