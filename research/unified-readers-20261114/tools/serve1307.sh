#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1307): memra-server for Qwen3.8-27B (GGUF
# NVFP4 trunk, Q5_K output head) on 127.0.0.1:8090, in one of two modes:
#   mtp    - memra#1221's serve.sh serving: the masked regime MTP draft, 262,144 context, 32 sessions
#            (the draft-state measurement runs on this route's draft-graph captures);
#   dflash - the DFlash2 drafter (MEMRA_DSPARK_SPEC=1), 32,768 context, 4 sessions, prefix cache and
#            reuse pool off so every request primes cold (the pooled DFlash2 verify runs here).
# Every server line is timestamped (memra#1221's tsrun.py). A memory guard stops the server when
# MemAvailable stays under GUARD_KB (3 GiB by default; the short-memory cell lowers it to 1 GiB, as
# memra#1257's did) for two checks; MemAvailable at 1 Hz goes to $OUT/mem.jsonl.
# EXTRA="NAME=value ..." adds server settings on top of the mode's (the cohort-of-four cells raise
# the spec gate's band with it: the default admits speculation for two sessions at most).
# usage: OUT=dir BIN=dir MODE=mtp|dflash [GUARD_KB=n] [EXTRA="NAME=value ..."] serve1307.sh   (writes server.log, guard.log, server.pid)
set -u
OUT=${OUT:?out dir}; BIN=${BIN:?bin dir}; MODE=${MODE:?mtp or dflash}
GUARD_KB=${GUARD_KB:-3145728}
EXTRA_SET=${EXTRA:-}
H=$(cd "$(dirname "$0")" && pwd)
TS=$H/../../spec-capture-headroom-20261112/tools/tsrun.py
mkdir -p "$OUT"
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
Q=$HOME/models/memra-q38
G8=$Q/Qwen3.8-27B-NVFP4-MTP-GGUF
export MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:8090 MEMRA_KV_HOST_MB=0 MEMRA_TICK_TRACE=1
case $MODE in
  mtp)
    export MEMRA_VISION_DIR=$HOME/models/Qwen3.8-27B-NVFP4 MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=32
    export MEMRA_MODELS="q38=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
    export MEMRA_MTP_DRAFT="$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"
    export MEMRA_PREFIX_STABLE_BOUNDARY=1 ;;
  dflash)
    export MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=4 MEMRA_PREFIX_CACHE_MB=0 MEMRA_REUSE_POOL=0
    export MEMRA_MODELS="q38=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
    export MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$Q/Qwen3.8-27B-DFlash2-memra ;;
  *) echo "MODE $MODE unknown" >&2; exit 2 ;;
esac
for kv in $EXTRA_SET; do export "$kv"; done
G=$OUT/guard.log
{ echo "$(date '+%F %T') start MODE=$MODE GUARD_KB=$GUARD_KB BIN=${BIN/#$HOME/~}"; sha256sum "$BIN/memra-server" | sed "s|$HOME|~|";
  env | grep '^MEMRA_' | sed "s|$HOME|~|g" | sort; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$G"
python3 "$TS" "$BIN/memra-server" > "$OUT/server.log" 2>&1 &
pid=$!
echo $pid > "$OUT/server.pid"
low=0; minv=999999999
while kill -0 $pid 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  printf '{"t":%s,"mem_available_kb":%s}\n' "$(date +%s.%N | cut -c1-14)" "$a" >> "$OUT/mem.jsonl"
  [ "$a" -lt "$minv" ] && minv=$a
  if [ "$a" -lt "$GUARD_KB" ]; then low=$((low+1)); else low=0; fi
  if [ $low -ge 2 ]; then echo "$(date '+%T') GUARD_KILL MemAvailable ${a}kB < ${GUARD_KB}kB twice" >> "$G"; kill -TERM $pid; break; fi
  sleep 1
done
wait $pid; rc=$?
echo "$(date '+%F %T') server exit rc=$rc min MemAvailable ${minv}kB" >> "$G"
