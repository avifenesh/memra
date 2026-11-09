#!/usr/bin/env bash
# One arm on the local RTX 5090: boot memra-server, run one workload, stop it.
#   arm.sh LABEL BIN OUTDIR WORKLOAD [VAR=VALUE ...]
# WORKLOAD: mixed | agentic:U[:TEMP[:MAXTOK[:prime]]]
# Qwen3.8-27B NVFP4+Q5_K GGUF (MTP head), MEMRA_CTX=32768, the prefix-budget-20261107 lane's shape.
# The server runs at the naked defaults plus the given VAR=VALUE pairs. Telemetry: nvidia-smi every
# 250 ms (device free MiB, utilization), /proc/meminfo MemAvailable every second, and every server
# stderr line prefixed with its unix time.
set -u
LABEL=$1; BIN=$2; OUT=$3; WL=$4; shift 4
# Yield seam: a YIELD file in the lane dir stops a phase at the next arm boundary.
if [ -e <lane>/YIELD ]; then echo "yield: $LABEL not run"; exit 0; fi
M=<models>/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
H=<harness>
WORKLOAD=<agentic-workload>
PORT=18287
D=$OUT/$LABEL; mkdir -p "$D"
{
  echo "arm $LABEL workload=$WL start=$(TZ=Asia/Jerusalem date -Is) profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"
  sha256sum "$BIN"
  echo "env: $*"
  nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm,power.limit,memory.used,memory.free --format=csv,noheader
} > "$D/arm.txt"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,temperature.gpu,clocks.sm,power.draw,utilization.gpu --format=csv,noheader -lms 250 > "$D/gpu-tel.csv" 2>&1 &
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
  "$@" "$BIN" 2>&1 < /dev/null | python3 -u "$H/ts.py" > "$D/serve.log" &
SP=$!
rc=0
# Stop waiting as soon as the server is gone (the log pipe's reader exits with it).
waitfor --until "curl -sf -o /dev/null http://127.0.0.1:$PORT/v1/models || ! kill -0 $SP" --timeout 8m --every 3s --quiet
if ! curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
  echo "boot FAILED" >> "$D/arm.txt"; tail -n 20 "$D/serve.log" >> "$D/arm.txt"; rc=3
else
  case "$WL" in
    mixed)
      timeout 900 python3 "$H/mixed.py" "$PORT" "$D/rows.json" > "$D/result.json" 2> "$D/client.err" || rc=$?
      ;;
    agentic:*)
      IFS=: read -r _ U T MT PO <<< "$WL"
      extra=""; [ "${PO:-}" = prime ] && extra="--prime-only"
      timeout 3600 python3 "$H/agentic2.py" "$WORKLOAD" "$D/rows.json" --port "$PORT" --users "$U" \
        --temperature "${T:-1.0}" --max-tokens "${MT:-768}" $extra > "$D/result.json" 2> "$D/client.err" || rc=$?
      ;;
    *) echo "unknown workload $WL" >> "$D/arm.txt"; rc=2 ;;
  esac
fi
# The server is the pipeline's first process; stop it by its listening port's owner.
SPID=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -n 1)
[ -n "$SPID" ] && kill -TERM "$SPID" 2>/dev/null
[ -n "$SPID" ] && { waitfor --gone "$SPID" --timeout 90s --quiet || kill -KILL "$SPID" 2>/dev/null; }
wait $SP 2>/dev/null
kill $TEL $MEMW 2>/dev/null; wait $TEL $MEMW 2>/dev/null
python3 - "$D" >> "$D/arm.txt" <<'EOF'
import sys, os
d = sys.argv[1]
free, util = [], []
for l in open(os.path.join(d, "gpu-tel.csv")):
    p = [x.strip() for x in l.split(",")]
    if len(p) >= 3 and p[2].endswith("MiB"):
        free.append(int(p[2].split()[0]))
    if len(p) >= 7 and p[6].endswith("%"):
        util.append(int(p[6].split()[0]))
mem = [int(l.split()[1]) for l in open(os.path.join(d, "memavail.txt")) if l.strip()]
print("device_free_min_mib", min(free) if free else None, "samples", len(free))
print("memavailable_min_gib %.2f" % (min(mem) / 1048576) if mem else "memavailable none")
EOF
grep -a -m1 '\[prefix-cache\] on:' "$D/serve.log" | cut -c1-600 >> "$D/arm.txt"
grep -a -c 'evict' "$D/serve.log" | sed 's/^/evict_lines=/' >> "$D/arm.txt"
grep -a -ci 'out of memory\|CUDA_ERROR_OUT_OF_MEMORY' "$D/serve.log" | sed 's/^/oom_lines=/' >> "$D/arm.txt"
grep -a -c 'reclaim-on-defer' "$D/serve.log" | sed 's/^/reclaim_on_defer_lines=/' >> "$D/arm.txt"
grep -a -c 'VRAM defer' "$D/serve.log" | sed 's/^/vram_defer_lines=/' >> "$D/arm.txt"
grep -a -c 'admit-trim' "$D/serve.log" | sed 's/^/admit_trim_lines=/' >> "$D/arm.txt"
cat "$D/result.json" >> "$D/arm.txt" 2>/dev/null
echo "rc=$rc end=$(TZ=Asia/Jerusalem date -Is)" >> "$D/arm.txt"
exit $rc
