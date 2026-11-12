#!/usr/bin/env bash
# Runs ON the GB10 host: memra-server for Qwen3.8-27B (GGUF NVFP4+Q5_K trunk + masked MTP draft,
# the recommended GB10 arm of research/gb10-tuning-20261102/tools/serve-arm.sh) with
#   - a memory guard (SIGTERM when MemAvailable < 3 GiB on two checks 1 s apart),
#   - MemAvailable at 1 Hz -> $OUT/mem.jsonl,
#   - GPU telemetry at 250 ms -> $OUT/gpu.csv.
# usage: OUT=dir BIN=dir [EXTRA="K=V ..."] [WRAP="nsys profile ..."] serve.sh
# Writes $OUT/server.log, $OUT/guard.log, $OUT/server.pid; exits when the server exits.
set -u
OUT=${OUT:?out dir}
BIN=${BIN:?bin dir}
mkdir -p "$OUT"
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
NV=$HOME/models/Qwen3.8-27B-NVFP4
export MEMRA_COMPAT=openai
export MEMRA_VISION_DIR="$NV"
export MEMRA_ADDR=127.0.0.1:8090
export MEMRA_CTX=262144
export MEMRA_MAX_SESSIONS=32
export MEMRA_KV_HOST_MB=0
export MEMRA_MODELS="q38=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
export MEMRA_MTP_DRAFT="$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"
export MEMRA_PREFIX_STABLE_BOUNDARY=1
for kv in ${EXTRA:-}; do export "$kv"; done
G=$OUT/guard.log
{ echo "$(date '+%F %T') start BIN=$BIN EXTRA=${EXTRA:-} WRAP=${WRAP:-}"; sha256sum "$BIN/memra-server";
  env | grep '^MEMRA_' | sort; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv; } > "$G"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 250 > "$OUT/gpu.csv" 2>&1 &
smi=$!
${WRAP:-} "$BIN/memra-server" > "$OUT/server.log" 2>&1 &
pid=$!
echo $pid > "$OUT/server.pid"
low=0; minv=999999999
while kill -0 $pid 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  printf '{"t":%s,"mem_available_kb":%s}\n' "$(date +%s.%N | cut -c1-14)" "$a" >> "$OUT/mem.jsonl"
  [ "$a" -lt "$minv" ] && minv=$a
  if [ "$a" -lt "${GUARD_KB:-3145728}" ]; then low=$((low+1)); else low=0; fi
  if [ $low -ge 2 ]; then echo "$(date '+%T') GUARD_KILL MemAvailable ${a}kB < ${GUARD_KB:-3145728}kB twice" >> "$G"; kill -TERM $pid; break; fi
  sleep 1
done
wait $pid; rc=$?
kill $smi 2>/dev/null
echo "$(date '+%F %T') server exit rc=$rc min MemAvailable ${minv}kB" >> "$G"
