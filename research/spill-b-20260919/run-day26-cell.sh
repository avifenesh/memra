#!/usr/bin/env bash
# Day 26 memra#539 cell (DAY26.md section 2): allocated versus used KV per request under the fixed mix, one order per
# invocation, no engine change. Shadow predictive receipts on (MEMRA_ADMIT_PREDICT_SHADOW=1): nothing is refused by them.
# usage: run-day26-cell.sh <cell-name> <AB|BA> [bin]     (cell dir: $RIGDIR/<cell-name>/)
# env: MODEL (artifact), MODEL_KEY (served name), RIGDIR (receipt root), LOCK (/tmp/memra-5090.lock; "none" when the
#      collector already holds the canonical lock), MEMRA_CTX (unset = the checkpoint's declared context), N (reps).
#      Day 27: MEMRA_KV_PARK_COMPACT and MEMRA_SERVE_SPEC pass through to the server (recorded in shape.txt); the
#      client records a sha256 per completion so arms compare byte-for-byte (day27-compare.py).
#      Day 31: CLIENT (default day26-client.py), CLIENT_ARGS (extra client flags) and PARSER (default day26-parse.py)
#      select the day-31 workload; the door env (MEMRA_ADMIT_BY_MEMORY, MEMRA_ADMIT_OPEN_OUTPUT_TOKENS) and
#      MEMRA_TIMEOUT_MS_MAX pass through to the server and are recorded in shape.txt. Unset, every default is day 26's.
set -uo pipefail
cell=${1:?cell}; order=${2:?AB|BA}
WT=${WT:-$HOME/projects/wt-spill-b}
RIGDIR=${RIGDIR:-$WT/research/spill-b-20260919/rtx5090-day26}
BIN=${3:-$WT/target/release/memra-server}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
MODEL_KEY=${MODEL_KEY:-q9}
PORT=${PORT:-18526}; BASE=http://127.0.0.1:$PORT
LOCK=${LOCK:-/tmp/memra-5090.lock}
N=${N:-5}
if command -v systemd-run >/dev/null 2>&1 && [ "${NO_SCOPE:-0}" != 1 ]; then
  run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
else
  run() { "$@"; }
fi
cd "$WT" || exit 1
[ -f "$MODEL" ] || { echo "model absent: $MODEL"; exit 2; }
C=$RIGDIR/$cell; rm -rf "$C"; mkdir -p "$C"
sha256sum "$BIN" > "$C/binary.sha256"; git rev-parse HEAD > "$C/source.txt"; git status --short > "$C/dirty.txt"
sha256sum "$MODEL" > "$C/model.sha256" &
echo "order=$order n=$N model_key=$MODEL_KEY ctx=${MEMRA_CTX:-unset} lock=$LOCK warm=${WARM_IMMEDIATE:+immediate}${WARM_IMMEDIATE:-deferred} park_compact=${MEMRA_KV_PARK_COMPACT:-unset} serve_spec=${MEMRA_SERVE_SPEC:-default}${CLIENT:+ client=$CLIENT args=${CLIENT_ARGS:-none} admit_by_memory=${MEMRA_ADMIT_BY_MEMORY:-unset} open_output_tokens=${MEMRA_ADMIT_OPEN_OUTPUT_TOKENS:-unset} timeout_ms_max=${MEMRA_TIMEOUT_MS_MAX:-unset} max_sessions=${MEMRA_MAX_SESSIONS:-unset} kv_host_mb=${MEMRA_KV_HOST_MB:-unset}}" > "$C/shape.txt"
[ -n "${CLIENT:-}" ] && sha256sum docs/SERVING.md "research/spill-b-20260919/$CLIENT" "research/spill-b-20260919/${PARSER:-day26-parse.py}" > "$C/inputs.sha256"
if [ "$LOCK" != none ]; then
  exec 9>"$LOCK"
  echo "$(date -u +%FT%TZ) waiting for $LOCK (flock -w 3600)" | tee "$C/lock.txt"
  flock -w 3600 9 || { echo "$(date -u +%FT%TZ) lock not acquired within 3600 s; cell not run" | tee -a "$C/lock.txt"; exit 3; }
  echo "$(date -u +%FT%TZ) lock acquired" | tee -a "$C/lock.txt"
else
  echo "$(date -u +%FT%TZ) lock held by the collector (external)" | tee "$C/lock.txt"
fi
wait
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$C/compute-apps-before.csv"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv > "$C/gpu-before.csv"
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 4; }
date -u +%FT%TZ > "$C/started.txt"
# server stderr is stamped per line (epoch ms) so log events align with the 250 ms samples.
MEMRA_COMPAT=openai MEMRA_MODELS="$MODEL_KEY=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT MEMRA_ADMIT_PREDICT_SHADOW=1 \
  run "$BIN" 2>&1 | python3 -u -c 'import sys,time
for line in sys.stdin:
    sys.stdout.write(f"{int(time.time()*1000)} {line}")' > "$C/server.log" &
SPID=$!
# identify THIS cell's server by cwd (this worktree) among memra-server processes; never touch one with another cwd.
own_server_pid() { for p in $(pgrep -x memra-server); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$WT" ] && echo $p; done; }
stop_server() { for p in $(own_server_pid); do kill -TERM $p 2>/dev/null; done; wait $SPID 2>/dev/null; }
trap stop_server EXIT
ready_ms=""
for _ in $(seq 1200); do
  if curl -sf -m 2 "$BASE/readyz" >/dev/null 2>&1; then ready_ms=$(python3 -c "import time; print(int(time.time()*1000))"); break; fi
  grep -q "FATAL\|panicked" "$C/server.log" && break
  sleep 0.5
done
[ -n "$ready_ms" ] || { echo "server never ready"; tail -20 "$C/server.log"; echo 5 > "$C.exit"; exit 5; }
echo "$ready_ms" > "$C/ready_ms.txt"
curl -s -m 5 "$BASE/metrics" > "$C/metrics-ready.json"
run python3 "$WT/research/spill-b-20260919/${CLIENT:-day26-client.py}" --base "$BASE" --out "$C" --model "$MODEL_KEY" \
  --order "$order" --n "$N" ${WARM_IMMEDIATE:+--warm-immediate} ${CLIENT_ARGS:-} > "$C/client.log" 2>&1
rc=$?; echo $rc > "$C.exit"
curl -s -m 5 "$BASE/metrics" > "$C/metrics-end.json"
stop_server; trap - EXIT
sleep 2
date -u +%FT%TZ > "$C/finished.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$C/compute-apps-after.csv"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv > "$C/gpu-after.csv"
python3 "$WT/research/spill-b-20260919/${PARSER:-day26-parse.py}" "$C" > "$C/REPORT.txt" 2>&1
tail -40 "$C/REPORT.txt"
exit $rc
