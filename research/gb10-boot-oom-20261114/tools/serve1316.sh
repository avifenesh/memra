#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): memra-server for Qwen3.8-27B (GGUF
# NVFP4 trunk, Q5_K output head) with the DFlash2 drafter on 127.0.0.1:8090, the serving both
# occurrences booted: memra#1307's serve1307.sh in its dflash mode (32,768 context, 4 sessions,
# prefix cache and reuse pool off). Every server line is timestamped (memra#1221's tsrun.py). A
# memory guard stops the server when MemAvailable stays under 3 GiB for two checks.
# usage: OUT=dir BIN=dir serve1316.sh   (writes server.log, guard.log, server.pid)
set -u
OUT=${OUT:?out dir}; BIN=${BIN:?bin dir}
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
. "$H/serve_env.sh"
G=$OUT/guard.log
{ echo "$(date '+%F %T') start BIN=${BIN/#$HOME/\~}"; sha256sum "$BIN/memra-server" | sed "s|$HOME|~|";
  env | grep '^MEMRA_' | sed "s|$HOME|~|g" | sort; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$G"
python3 "$H/../../spec-capture-headroom-20261112/tools/tsrun.py" "$BIN/memra-server" > "$OUT/server.log" 2>&1 &
pid=$!
echo $pid > "$OUT/server.pid"
low=0
while kill -0 $pid 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$a" -lt 3145728 ]; then low=$((low+1)); else low=0; fi
  if [ $low -ge 2 ]; then echo "$(date '+%T') GUARD_KILL MemAvailable ${a}kB twice" >> "$G"; kill -TERM $pid; break; fi
  sleep 1
done
wait $pid; rc=$?
echo "$(date '+%F %T') server exit rc=$rc" >> "$G"
