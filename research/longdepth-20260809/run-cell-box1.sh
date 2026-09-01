#!/usr/bin/env bash
# One bounded matrix cell: boot one server, issue N fresh requests, analyze, stop, unlock.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH

ROOT=$(git rev-parse --show-toplevel)
LANE=$ROOT/research/longdepth-20260809
RUN_ID=${LONGDEPTH_RUN_ID:?set LONGDEPTH_RUN_ID once for the matrix}
RUN=$LANE/raw/$RUN_ID
CTX=${LONGDEPTH_CTX:?set LONGDEPTH_CTX to 262144}
DEPTH=${LONGDEPTH_DEPTH:?set LONGDEPTH_DEPTH to 2048, 6144, or 12288}
MODE=${LONGDEPTH_SPEC_MODE:?set LONGDEPTH_SPEC_MODE to on or off}
TEMP=${LONGDEPTH_TEMP:?set LONGDEPTH_TEMP to 0 or 0.7}
TOP_P=${LONGDEPTH_TOP_P:-1}
SAMPLE_BACKEND=${LONGDEPTH_SAMPLE_BACKEND:-gpu}
VARIANT=${LONGDEPTH_VARIANT:-baseline}
REPS=${LONGDEPTH_REPS:-2}
PORT=${LONGDEPTH_PORT:-18209}
MODEL=${MEMRA_STEP37_GGUF:-$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${MEMRA_STEP37_DRAFT:-$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf}
BIN=$ROOT/target/release/memra-server
TOK_SPAN=$ROOT/target/release/tok_span
PYTHON=${LONGDEPTH_PYTHON:-/tmp/memra-longdepth-parser-$RUN_ID/bin/python3}
TEMP_LABEL=${TEMP/./p}
TOP_P_LABEL=${TOP_P/./p}
if [[ "$VARIANT" == baseline ]]; then
  CELL=ctx${CTX}-${MODE}-t${TEMP_LABEL}-d${DEPTH}
else
  CELL=ctx${CTX}-${MODE}-t${TEMP_LABEL}-p${TOP_P_LABEL}-${SAMPLE_BACKEND}-d${DEPTH}-${VARIANT}
fi
OUT=$LANE/raw/$RUN_ID/cells/$CELL
BASE=http://127.0.0.1:$PORT

case "$CTX" in 262144) ;; *) printf 'invalid ctx: %s (steering pins 262144)\n' "$CTX" >&2; exit 2 ;; esac
case "$DEPTH" in 2048|6144|12288) ;; *) printf 'invalid depth: %s\n' "$DEPTH" >&2; exit 2 ;; esac
case "$MODE" in on|off) ;; *) printf 'invalid mode: %s\n' "$MODE" >&2; exit 2 ;; esac
case "$TEMP" in 0|0.7) ;; *) printf 'invalid temperature: %s\n' "$TEMP" >&2; exit 2 ;; esac
case "$TOP_P" in 1|0.9) ;; *) printf 'invalid top_p: %s\n' "$TOP_P" >&2; exit 2 ;; esac
case "$SAMPLE_BACKEND" in gpu|host) ;; *) printf 'invalid sample backend: %s\n' "$SAMPLE_BACKEND" >&2; exit 2 ;; esac
if [[ ! "$VARIANT" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  printf 'invalid variant label: %s\n' "$VARIANT" >&2
  exit 2
fi
if [[ "$SAMPLE_BACKEND" == host && "$MODE" != off ]]; then
  printf 'host sampler oracle is valid only with speculative mode off\n' >&2
  exit 2
fi
test -x "$BIN"
test -x "$TOK_SPAN"
test -x "$PYTHON"
test -f "$MODEL"
test -f "$DRAFT"
test -f "$RUN/rendered-prompt-low.txt"
if [[ -e "$OUT" ]]; then
  printf 'refusing to overwrite existing cell: %s\n' "$OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
cd "$ROOT"

SERVER_PID=
stop_server() {
  if [[ -n ${SERVER_PID:-} ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
        return
      fi
      sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}
trap stop_server EXIT

exec 9>/tmp/memra-gpu.lock
flock -w 7200 9 || { printf 'GPU lock timeout for %s\n' "$CELL" >&2; exit 75; }
printf 'lock acquired %s cell=%s\n' "$(date -u +%FT%TZ)" "$CELL" | tee "$OUT/driver.log"
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv,noheader > "$OUT/gpu-pre.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
  --format=csv,noheader > "$OUT/gpu-processes-pre.csv" 2>&1 || true

if ss -tln 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"; then
  printf 'port %s already in use\n' "$PORT" | tee -a "$OUT/driver.log" >&2
  exit 1
fi

spec_env=()
if [[ "$MODE" == on ]]; then
  spec_env=(MEMRA_SERVE_SPEC=1 MEMRA_SPEC_GATE=0)
else
  spec_env=(MEMRA_SERVE_SPEC=0)
fi
sample_env=()
if [[ "$SAMPLE_BACKEND" == host ]]; then
  sample_env=(MEMRA_SERVE_DEVSAMPLE=0)
fi
command=(
  env
  -u MEMRA_COMPAT
  -u MEMRA_API_KEY
  -u MEMRA_API_KEYS
  -u MEMRA_SERVE_SPEC
  -u MEMRA_SPEC_GATE
  -u MEMRA_SPEC_GATE_LOW
  -u MEMRA_SPEC_GATE_HIGH
  -u MEMRA_SPEC_K
  -u MEMRA_PRIME_CHUNK
  -u MEMRA_PREFILL_TICK
  -u MEMRA_STEP35_SWA_TKV
  -u MEMRA_PRIME_CALLLOCAL
  -u MEMRA_STEP35_SWA_FA
  -u MEMRA_NOFA
  -u MEMRA_SERVE_DEVSAMPLE
  "${spec_env[@]}"
  "${sample_env[@]}"
  "MEMRA_MODELS=step35=${MODEL}+${DRAFT}"
  MEMRA_PP_STAGES=2
  "MEMRA_PP_DEVICES=0,1"
  MEMRA_MOE_GROUPED=1
  "MEMRA_CTX=$CTX"
  MEMRA_KV_REUSE=0
  MEMRA_REUSE_POOL=0
  MEMRA_PREFIX_CACHE_MB=0
  "MEMRA_ADDR=127.0.0.1:$PORT"
  "$BIN"
)
{
  printf 'cell=%s\n' "$CELL"
  printf 'started=%s\n' "$(date -u +%FT%TZ)"
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  printf 'mode=%s ctx=%s depth=%s temp=%s top_p=%s sample_backend=%s variant=%s reps=%s\n' \
    "$MODE" "$CTX" "$DEPTH" "$TEMP" "$TOP_P" "$SAMPLE_BACKEND" "$VARIANT" "$REPS"
  printf 'command:'
  printf ' %q' "${command[@]}"
  printf '\n'
} >> "$OUT/driver.log"

"${command[@]}" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 240); do
  if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'server died before ready\n' | tee -a "$OUT/driver.log" >&2
    tail -100 "$OUT/server.log" | tee -a "$OUT/driver.log" >&2
    exit 1
  fi
  sleep 2
done
curl -sf "$BASE/readyz" > "$OUT/readyz.json"
if ! ss -tlnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" | grep -q "pid=$SERVER_PID,"; then
  printf 'port responder is not owned server pid=%s\n' "$SERVER_PID" | tee -a "$OUT/driver.log" >&2
  exit 1
fi

for rep in $(seq 1 "$REPS"); do
  REP_OUT=$OUT/rep$rep
  "$PYTHON" "$LANE/request.py" \
    --base "$BASE" \
    --model step35 \
    --prompt-file "$RUN/rendered-prompt-low.txt" \
    --max-tokens "$DEPTH" \
    --temperature "$TEMP" \
    --top-p "$TOP_P" \
    --cell "$CELL" \
    --rep "$rep" \
    --out-dir "$REP_OUT" \
    --timeout 3600 \
    2>&1 | tee "$OUT/request-rep${rep}.log"
  "$PYTHON" "$LANE/detect.py" \
    --response "$REP_OUT/response.json" \
    --model "$MODEL" \
    --tok-span "$TOK_SPAN" \
    --out "$REP_OUT/detector.json" \
    --label "$CELL-r$rep" \
    --doctype-prefilled \
    2>&1 | tee "$OUT/detector-rep${rep}.log"
done

curl -sf "$BASE/metrics" > "$OUT/metrics.txt" 2>&1 || true
stop_server
sleep 3
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv,noheader > "$OUT/gpu-post.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
  --format=csv,noheader > "$OUT/gpu-processes-post.csv" 2>&1 || true
SPEC_ACC_LINES=$(grep -c '\[spec-acc\]' "$OUT/server.log" || true)
CELL_STATUS=0
if [[ "$MODE" == on && "$SPEC_ACC_LINES" == 0 ]]; then
  printf 'spec execution not confirmed: forced-on cell has zero [spec-acc] lines\n' \
    | tee -a "$OUT/driver.log" >&2
  CELL_STATUS=1
elif [[ "$MODE" == off && "$SPEC_ACC_LINES" != 0 ]]; then
  printf 'spec disablement violated: off cell has %s [spec-acc] lines\n' "$SPEC_ACC_LINES" \
    | tee -a "$OUT/driver.log" >&2
  CELL_STATUS=1
fi
{
  printf 'spec_acc_lines=%s\n' "$SPEC_ACC_LINES"
  printf 'finished=%s\n' "$(date -u +%FT%TZ)"
  printf 'lock released cell=%s\n' "$CELL"
} | tee -a "$OUT/driver.log"
flock -u 9
trap - EXIT
exit "$CELL_STATUS"
