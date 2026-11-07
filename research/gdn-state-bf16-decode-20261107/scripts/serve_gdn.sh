#!/usr/bin/env bash
# Runs ON a GB10 unit: memra-server for Qwen3.8-27B (GGUF NVFP4+Q5_K trunk, masked sxc32768 MTP
# draft: the `gguf-mtpmask` arm of the batch-spec research harness), under a memory guard
# (killed if MemAvailable stays under 3 GiB for two checks 2 s apart). Derived unified-memory
# defaults, no hand-set prefix or TTFT knobs. The GDN decode class is the only arm variable:
#   BIN=<dir with memra-server> LEASE=<sparklock lease> CLASS=0|1|default TAG=<name> serve_gdn.sh
# Logs: ~/.cache/memra-gdnbf16-decode/runs/serve-$TAG.{log,guard,pid}
set -u
BIN=${BIN:?bin dir}
TAG=${TAG:?tag}
CLASS=${CLASS:?class}
D=$HOME/.cache/memra-gdnbf16-decode/runs
mkdir -p "$D"
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
NV=$HOME/models/Qwen3.8-27B-NVFP4
export SPARKLOCK_LEASE=${LEASE:-}
export MEMRA_COMPAT=openai
export MEMRA_VISION_DIR="$NV"
export MEMRA_ADDR=127.0.0.1:8090
export MEMRA_CTX=262144
export MEMRA_MAX_SESSIONS=32
export MEMRA_KV_HOST_MB=0
export MEMRA_MODELS="q38=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
export MEMRA_MTP_DRAFT="$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"
export MEMRA_PREFIX_STABLE_BOUNDARY=1
case "$CLASS" in
  0|1) export MEMRA_GDN_STATE_BF16=$CLASS ;;
  default) unset MEMRA_GDN_STATE_BF16 ;;
  *) echo "CLASS must be 0, 1 or default" >&2; exit 2 ;;
esac
GD=$D/serve-$TAG.guard
{ echo "$(date '+%F %T') start TAG=$TAG CLASS=$CLASS BIN=$BIN"; sha256sum "$BIN/memra-server"; env | grep '^MEMRA_' | sort; } > "$GD"
"$BIN/memra-server" > "$D/serve-$TAG.log" 2>&1 &
pid=$!
echo $pid > "$D/serve-$TAG.pid"
low=0; minv=999999999
while kill -0 $pid 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$a" -lt "$minv" ] && minv=$a
  if [ "$a" -lt 3145728 ]; then low=$((low+1)); else low=0; fi
  if [ $low -ge 2 ]; then echo "$(date '+%T') GUARD_KILL MemAvailable ${a}kB < 3GiB twice" >> "$GD"; kill -TERM $pid; sleep 5; kill -KILL $pid 2>/dev/null; break; fi
  if [ $((SECONDS % 300)) -lt 2 ]; then echo "$(date '+%T') MemAvailable now ${a}kB min ${minv}kB" >> "$GD"; fi
  sleep 2
done
wait $pid; rc=$?
echo "$(date '+%F %T') server exit rc=$rc min MemAvailable ${minv}kB" >> "$GD"
