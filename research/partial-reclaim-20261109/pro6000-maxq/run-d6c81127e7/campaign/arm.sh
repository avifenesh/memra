#!/usr/bin/env bash
# One partial-reclaim arm on one RTX PRO 6000 Max-Q: boot memra-server, run one workload, stop it.
# Copy of the #1090 lane's arm-limit.sh with this lane's harness dir and TMPDIR.
#   arm.sh LABEL BIN OUTDIR WORKLOAD [VAR=VALUE ...]
# WORKLOAD: mixed | agentic:U[:TEMP[:MAXTOK]]
# Port of research/prefix-budget-20261107/harness/arm.sh (the 5090 lane's arm): same model, MEMRA_CTX=32768,
# naked defaults plus the given VAR=VALUE pairs, nvidia-smi every 250 ms on the pinned card (device free
# MiB), /proc/meminfo MemAvailable every second. The card is pinned by UUID.
set -u
. <kit>/env.sh
export TMPDIR=<lane>/tmp
LABEL=$1; BIN=$2; OUT=$3; WL=$4; shift 4
MODEL=$M/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
H=<lane>/harness
AH=$W/agentic/home/avifenesh/.cache/model-scout/harness
WORK=$W/agentic/home/avifenesh/.cache/memra-1025/ab-workload-5090.json
GPU=${ARM_GPU_UUID:-$GPU3_UUID}
PORT=${ARM_PORT:-18287}
D=$OUT/$LABEL; mkdir -p "$D"
{
  echo "arm $LABEL workload=$WL start=$(TZ=Asia/Jerusalem date -Is) gpu=$GPU"
  sha256sum "$BIN"
  echo "env: $*"
  nvidia-smi -i "$GPU" --query-gpu=index,name,temperature.gpu,fan.speed,clocks.sm,power.limit,memory.used,memory.free --format=csv,noheader
  echo "box: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader | tr '\n' ';')"
} > "$D/arm.txt"
nvidia-smi -i "$GPU" --query-gpu=timestamp,memory.used,memory.free,temperature.gpu,clocks.sm,power.draw --format=csv,noheader -lms 250 > "$D/gpu-tel.csv" 2>&1 &
TEL=$!
python3 -c '
import time, sys
f = open(sys.argv[1], "w")
while True:
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            f.write("%d %s\n" % (time.time(), l.split()[1])); f.flush()
    time.sleep(1)
' "$D/memavail.txt" &
MEMW=$!
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
  "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=32768 "$@" "$BIN" > "$D/serve.log" 2>&1 &
SP=$!
rc=0
if ! waitfor --http "http://127.0.0.1:$PORT/v1/models" --timeout 8m --every 3s --fail-after 160 --quiet; then
  echo "boot FAILED" >> "$D/arm.txt"; tail -n 20 "$D/serve.log" >> "$D/arm.txt"; rc=3
else
  case "$WL" in
    mixed)
      timeout 900 python3 "$H/mixed.py" "$PORT" "$D/rows.json" > "$D/result.json" 2> "$D/client.err" || rc=$?
      ;;
    limit)
      timeout 3000 python3 "$H/limit.py" "$PORT" "$D/rows.json" > "$D/result.json" 2> "$D/client.err" || rc=$?
      ;;
    agentic:*)
      IFS=: read -r _ U T MT <<< "$WL"
      (cd "$AH" && CLIENT_MAX_TIME=1800 BENCH_BASE=http://127.0.0.1:$PORT \
        BENCH_MODEL=gate BENCH_TOKENIZE_PATH=/v1/tokenize BENCH_EMULATE_CONTINUOUS_USAGE=1 \
        timeout 3600 python3 "$H/agentic.py" "$WORK" \
        "$D/rows.json" --users "$U" --temperature "${T:-1.0}" --max-tokens "${MT:-768}") > "$D/result.json" 2> "$D/client.err" || rc=$?
      ;;
    *) echo "unknown workload $WL" >> "$D/arm.txt"; rc=2 ;;
  esac
fi
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > "$D/compute-apps-end.txt" 2>&1
kill -TERM $SP 2>/dev/null; waitfor --gone $SP --timeout 90s --quiet || kill -KILL $SP 2>/dev/null
wait $SP 2>/dev/null
kill $TEL $MEMW 2>/dev/null; wait $TEL $MEMW 2>/dev/null
python3 - "$D" >> "$D/arm.txt" <<'EOF'
import sys, os
d = sys.argv[1]
free = []
for l in open(os.path.join(d, "gpu-tel.csv")):
    p = [x.strip() for x in l.split(",")]
    if len(p) >= 3 and p[2].endswith("MiB"):
        free.append(int(p[2].split()[0]))
mem = [int(l.split()[1]) for l in open(os.path.join(d, "memavail.txt")) if l.strip()]
print("device_free_min_mib", min(free) if free else None, "samples", len(free))
print("memavailable_min_gib %.2f" % (min(mem) / 1048576) if mem else "memavailable none")
EOF
grep -a -m1 '\[prefix-cache\] on:' "$D/serve.log" | cut -c1-1200 >> "$D/arm.txt"
grep -a -c 'evict' "$D/serve.log" | sed 's/^/evict_lines=/' >> "$D/arm.txt"
grep -a -ci 'out of memory\|CUDA_ERROR_OUT_OF_MEMORY' "$D/serve.log" | sed 's/^/oom_lines=/' >> "$D/arm.txt"
grep -a -c 'reclaim-on-defer' "$D/serve.log" | sed 's/^/reclaim_on_defer_lines=/' >> "$D/arm.txt"
grep -a -c 'admit-oom' "$D/serve.log" | sed 's/^/admit_oom_lines=/' >> "$D/arm.txt"
grep -a -ci 'defer' "$D/serve.log" | sed 's/^/defer_lines=/' >> "$D/arm.txt"
cat "$D/result.json" >> "$D/arm.txt" 2>/dev/null
echo "rc=$rc end=$(TZ=Asia/Jerusalem date -Is)" >> "$D/arm.txt"
exit $rc
