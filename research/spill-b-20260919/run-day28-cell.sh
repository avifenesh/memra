#!/usr/bin/env bash
# Day 28 memra#476 shape-walk cell (DAY28.md 1.4): the two books, the device, and every pool grow on the local RTX 5090,
# the same script before and after the fix. Shadow mode only (MEMRA_ADMIT_PREDICT_SHADOW=1): nothing is refused.
# usage: run-day28-cell.sh <cell-name> [bin]     (cell dir: rtx5090-day28/<cell-name>/)
set -uo pipefail
cell=${1:?cell}
WT=${WT:-$HOME/projects/wt-spill-b}; R=$WT/research/spill-b-20260919/rtx5090-day28
BIN=${2:-$WT/target/release/memra-server}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
PORT=${PORT:-18478}; BASE=http://127.0.0.1:$PORT
LOCK=/tmp/memra-5090.lock
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
cd "$WT" || exit 1
[ -f "$MODEL" ] || { echo "model absent: $MODEL"; exit 2; }
C=$R/$cell; rm -rf "$C"; mkdir -p "$C"
sha256sum "$BIN" > "$C/binary.sha256"; git rev-parse HEAD > "$C/source.txt"; sha256sum "$MODEL" > "$C/model.sha256" &
exec 9>"$LOCK"
echo "$(date -u +%FT%TZ) waiting for $LOCK (flock -w 3600)" | tee "$C/lock.txt"
flock -w 3600 9 || { echo "$(date -u +%FT%TZ) lock not acquired within 3600 s; cell not run" | tee -a "$C/lock.txt"; exit 3; }
echo "$(date -u +%FT%TZ) lock acquired" | tee -a "$C/lock.txt"
wait
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$C/compute-apps-before.csv"
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,power.draw --format=csv > "$C/gpu-before.csv"
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 4; }
date -u +%FT%TZ > "$C/started.txt"
# server stderr is stamped per line (epoch ms) so log events align with the 250 ms samples.
MEMRA_COMPAT=openai MEMRA_MODELS="q9=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT MEMRA_CTX=65536 MEMRA_ADMIT_PREDICT_SHADOW=1 MEMRA_TTFT_TRACE=1 \
  run "$BIN" 2>&1 | python3 -u -c 'import sys,time
for line in sys.stdin:
    sys.stdout.write(f"{int(time.time()*1000)} {line}")' > "$C/server.log" &
SPID=$!
# the server is a child of systemd-run, not of the stamper pipeline: identify THIS cell's server by cwd (the lane's
# worktree) among memra-server processes started after this cell began; never touch one with another cwd.
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
run python3 "$WT/research/spill-b-20260919/day28-client.py" --base "$BASE" --out "$C" --model q9 > "$C/client.log" 2>&1
rc=$?; echo $rc > "$C.exit"
stop_server; trap - EXIT
sleep 2
date -u +%FT%TZ > "$C/finished.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$C/compute-apps-after.csv"
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,power.draw --format=csv > "$C/gpu-after.csv"
python3 "$WT/research/spill-b-20260919/day28-parse.py" "$C" > "$C/REPORT.txt" 2>&1
tail -25 "$C/REPORT.txt"
exit $rc
