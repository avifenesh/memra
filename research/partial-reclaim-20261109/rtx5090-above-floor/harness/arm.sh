#!/usr/bin/env bash
# One arm on the local RTX 5090 Laptop (24 GB): boot memra-server on Qwen3.5-9B NVFP4 with its MTP head at
# MEMRA_CTX=32768, run one workload, stop it. Port of research/prefix-budget-20261107/harness/arm.sh.
#   arm.sh LABEL BIN OUTDIR WORKLOAD [VAR=VALUE ...]
# WORKLOAD: probe (boot, read the budget line, stop) | limit (harness/limit.py with $LIMIT_ARGS) | mixed
# Telemetry: nvidia-smi every 250 ms (device free MiB), /proc/meminfo MemAvailable every second.
set -u
LABEL=$1; BIN=$2; OUT=$3; WL=$4; shift 4
M=${ARM_MODEL:-<models>/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
H=$(cd "$(dirname "$0")/../../../prefix-budget-20261107/harness" && pwd)
PORT=${ARM_PORT:-18197}
D=$OUT/$LABEL; mkdir -p "$D"
{
  echo "arm $LABEL workload=$WL start=$(TZ=Asia/Jerusalem date -Is) profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"
  sha256sum "$BIN"
  echo "env: $* LIMIT_ARGS=${LIMIT_ARGS:-}"
  nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm,power.limit,memory.used,memory.free --format=csv,noheader
} > "$D/arm.txt"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,temperature.gpu,clocks.sm,power.draw --format=csv,noheader -lms 250 > "$D/gpu-tel.csv" 2>&1 &
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
env MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$M" "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=32768 \
  "$@" "$BIN" > "$D/serve.log" 2>&1 &
SP=$!
rc=0
if ! waitfor --http "http://127.0.0.1:$PORT/v1/models" --timeout 8m --every 3s --fail-after 160 --quiet; then
  echo "boot FAILED" >> "$D/arm.txt"; tail -n 20 "$D/serve.log" >> "$D/arm.txt"; rc=3
else
  case "$WL" in
    probe) ;;
    limit)
      # shellcheck disable=SC2086
      timeout 3000 python3 "$H/limit.py" "$PORT" "$D/rows.json" ${LIMIT_ARGS:-} > "$D/result.json" 2> "$D/client.err" || rc=$?
      ;;
    mixed)
      timeout 900 python3 "$H/mixed.py" "$PORT" "$D/rows.json" > "$D/result.json" 2> "$D/client.err" || rc=$?
      ;;
    *) echo "unknown workload $WL" >> "$D/arm.txt"; rc=2 ;;
  esac
fi
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
grep -a -m1 '\[prefix-cache\] on:' "$D/serve.log" | cut -c1-900 >> "$D/arm.txt"
grep -a -c 'evict' "$D/serve.log" | sed 's/^/evict_lines=/' >> "$D/arm.txt"
grep -a -ci 'out of memory\|CUDA_ERROR_OUT_OF_MEMORY' "$D/serve.log" | sed 's/^/oom_lines=/' >> "$D/arm.txt"
grep -a -c 'reclaim-on-defer' "$D/serve.log" | sed 's/^/reclaim_on_defer_lines=/' >> "$D/arm.txt"
cat "$D/result.json" >> "$D/arm.txt" 2>/dev/null
echo "rc=$rc end=$(TZ=Asia/Jerusalem date -Is)" >> "$D/arm.txt"
exit $rc
