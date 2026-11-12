#!/usr/bin/env bash
# Laptop, inside a gpulock lease on the RTX 5090 (memra#1255): one boot of the agentic replay at
# c = 2 (USERS overrides), the shape of memra#1255 (from research/drafter-admission-20261112/
# harness/agentic-5090.sh): Qwen3.8-27B NVFP4 + Q5_K with the DFlash2 drafter and the FR-Spec
# trim, MEMRA_MTP_SKIP=1, ctx 32,768, 8 sessions, prefix cache 4,000 MB, no host tier; 29 agentic
# bodies (1.4k to 17k prompt tokens), T=1 seed 1234, thinking on, 768 tokens; a prime pass at
# max_tokens 1, then the timed pass.
# usage: agentic-5090.sh ARM    (binary bin/memra-server-ARM-120a). The power limit is recorded, never changed.
set -u
ARM=$1
S=$HOME/.cache/memra-1255
M=$HOME/models/q38-gguf
H=$S/harness
W=$HOME/.cache/memra-1025/ab-workload-5090.json
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
TAG=$ARM-$(date +%H%M%S)
D=$S/run/5090/${AG_SUB:-agentic}/$TAG
mkdir -p "$D"
if ss -ltn | grep -q ':18175 '; then echo "port 18175 busy; refusing"; exit 4; fi
{ date -Is; sha256sum "$S/bin/memra-server-$ARM-120a" "$W"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader
  uptime; } > "$D/guard.log"
setsid env MEMRA_COMPAT=openai MEMRA_MODELS="q38=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_ADDR=127.0.0.1:18175 \
  MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=8 MEMRA_KV_HOST_MB=0 MEMRA_PREFIX_CACHE_MB=4000 \
  MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$M/Qwen3.8-27B-DFlash2-memra \
  MEMRA_FRSPEC_TRIM=$M/q38-ranks-sxc32768.gguf.txt MEMRA_PREFIX_STABLE_BOUNDARY=0 \
  "$S/bin/memra-server-$ARM-120a" > "$D/serve.log" 2>&1 &
SP=$!
echo "$SP" > "$D/server.pgid"
echo "$SP" >> "$S/pids.txt"
trap 'kill -TERM -- -"$SP" 2>/dev/null; kill "$TP" 2>/dev/null; exit 143' TERM INT
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,utilization.gpu,memory.used \
  --format=csv,noheader -lms 250 > "$D/telemetry.csv" 2>/dev/null &
TP=$!
rc=0
if waitfor --http http://127.0.0.1:18175/v1/models --timeout 8m --every 5s --fail-after 96 --quiet && kill -0 "$SP" 2>/dev/null; then
  for u in ${USERS:-2}; do
    CLIENT_MAX_TIME=900 BENCH_BASE=http://127.0.0.1:18175 BENCH_MODEL=q38 timeout 1300 \
      python3 "$H/agent_turns.py" "$W" "$D" "$TAG" --users "$u" --think 1 --temperature 1.0 --seed 1234 \
      >> "$D/cells.log" 2>&1 || rc=$?
  done
  curl -s http://127.0.0.1:18175/metrics > "$D/metrics.json" 2>/dev/null
else
  echo "boot FAILED" | tee -a "$D/guard.log"; rc=3
fi
kill -TERM "$SP" 2>/dev/null; waitfor --gone "$SP" --timeout 90s --quiet || kill -KILL -- -"$SP" 2>/dev/null
kill "$TP" 2>/dev/null
{ echo "end $(date -Is) rc=$rc"
  for pat in 'drafter refused' 'eager-arm admit' 'VRAM defer' 'capacity reject' 'out of memory' 'settle wait' 'diag1255] gate-pass'; do
    echo "$pat=$(grep -a -c -F "$pat" "$D/serve.log")"
  done; } | tee -a "$D/guard.log"
tail -n 1 "$D/cells.log" | cut -c1-400
exit $rc
