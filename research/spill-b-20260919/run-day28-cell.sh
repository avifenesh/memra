#!/usr/bin/env bash
# Day 28 memra#476 shape-walk cell (DAY28.md 1.4): the two books, the device, and every pool grow, one fixed request
# sequence (warm, S1, S2, S4, S8, L, S8b). Shadow mode only (MEMRA_ADMIT_PREDICT_SHADOW=1): nothing is refused.
# usage: run-day28-cell.sh <cell-name> [bin]     (cell dir: $RIGDIR/<cell-name>/)
# env: WT (tree), RIGDIR (receipt root), MODEL (artifact), MODEL_KEY (served name), PORT, LOCK (/tmp/memra-5090.lock;
#      "none" when the collector already holds the canonical lock), NO_SCOPE=1 (no systemd-run scope), CTX (65536 by
#      default, the local shape; "unset" leaves MEMRA_CTX unset so the checkpoint's context serves, the target-card shape).
set -uo pipefail
cell=${1:?cell}
WT=${WT:-$HOME/projects/wt-spill-b}
RIGDIR=${RIGDIR:-$WT/research/spill-b-20260919/rtx5090-day28}
BIN=${2:-$WT/target/release/memra-server}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
MODEL_KEY=${MODEL_KEY:-q9}
PORT=${PORT:-18478}; BASE=http://127.0.0.1:$PORT
LOCK=${LOCK:-/tmp/memra-5090.lock}
CTX=${CTX:-65536}
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
echo "model_key=$MODEL_KEY ctx=$CTX lock=$LOCK shadow=1" > "$C/shape.txt"
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
# `run` is a shell function: env assignments must be shell prefixes, never `env ... run` (attempt 1 on the target card
# died on `env: 'run': No such file or directory`). MEMRA_CTX is exported only when CTX is a number.
if [ "$CTX" != unset ]; then export MEMRA_CTX=$CTX; else unset MEMRA_CTX; fi
MEMRA_COMPAT=openai MEMRA_MODELS="$MODEL_KEY=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT MEMRA_ADMIT_PREDICT_SHADOW=1 MEMRA_TTFT_TRACE=1 \
  run "$BIN" 2>&1 | python3 -u -c 'import sys,time
for line in sys.stdin:
    sys.stdout.write(f"{int(time.time()*1000)} {line}")' > "$C/server.log" &
SPID=$!
# identify THIS cell's server by cwd (the lane's tree) among memra-server processes; never touch one with another cwd.
own_server_pid() { for p in $(pgrep -x memra-server); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$WT" ] && echo $p; done; }
stop_server() { for p in $(own_server_pid); do kill -TERM $p 2>/dev/null; done; wait $SPID 2>/dev/null; }
trap stop_server EXIT
ready_ms=""
for _ in $(seq 600); do
  if curl -sf -m 2 "$BASE/readyz" >/dev/null 2>&1; then ready_ms=$(python3 -c "import time; print(int(time.time()*1000))"); break; fi
  grep -q "FATAL\|panicked" "$C/server.log" && break
  sleep 0.5
done
[ -n "$ready_ms" ] || { echo "server never ready"; tail -20 "$C/server.log"; echo 5 > "$C.exit"; exit 5; }
echo "$ready_ms" > "$C/ready_ms.txt"
run python3 "$WT/research/spill-b-20260919/day28-client.py" --base "$BASE" --out "$C" --model "$MODEL_KEY" > "$C/client.log" 2>&1
rc=$?; echo $rc > "$C.exit"
stop_server; trap - EXIT
sleep 2
date -u +%FT%TZ > "$C/finished.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$C/compute-apps-after.csv"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv > "$C/gpu-after.csv"
python3 "$WT/research/spill-b-20260919/day28-parse.py" "$C" > "$C/REPORT.txt" 2>&1
tail -12 "$C/REPORT.txt"
exit $rc
