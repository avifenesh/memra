#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1307): fill the page cache with four readers
# until MemFree is under LIMIT_MB (memra#1257's fill_cache2.sh), then keep READERS readers going
# over the model files, largest first, until STOP_FILE exists or KEEP_S seconds pass, so the page
# cache keeps filling and the kernel keeps reclaiming through the cell's requests. MemFree,
# MemAvailable and Cached at 1 Hz go to MEMLOG. Reads only; changes no setting.
# usage: live_fill.sh LIMIT_MB READERS KEEP_S STOP_FILE MEMLOG
set -u
LIM=$1; NR=$2; KEEP=$3; STOP=$4; MEMLOG=$5
H=$(cd "$(dirname "$0")" && pwd)
mi() { grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | awk '{printf "%s %d ", $1, $2/1024}'; }
(
  while [ ! -e "$STOP" ]; do echo "$(date +%s.%N | cut -c1-14) $(mi)"; sleep 1; done
) >> "$MEMLOG" 2>/dev/null < /dev/null &
echo "$(date -Is) live_fill: before $(mi)"
bash "$H/../../admit-headroom-unified-20261113/tools/fill_cache2.sh" "$LIM" 300 4 | sed 's/^/  /'
L=$(mktemp)
ls -S $HOME/models/*.gguf $HOME/models/*/*.gguf $HOME/models/*/*/*.gguf $HOME/models/*/*.safetensors \
  $HOME/models/*/*/*.safetensors 2>/dev/null > "$L"
pids=()
for i in $(seq 0 $((NR - 1))); do
  (
    end=$((SECONDS + KEEP))
    while [ ! -e "$STOP" ] && [ $SECONDS -lt $end ]; do
      awk -v n="$NR" -v i="$i" 'NR % n == i' "$L" | while read -r f; do
        [ -e "$STOP" ] && break
        [ $SECONDS -lt $end ] || break
        timeout 30 cat "$f" > /dev/null
      done
    done
  ) > /dev/null 2>&1 < /dev/null &
  pids+=($!)
done
echo "$(date -Is) live_fill: $NR readers running until $STOP (at most ${KEEP}s): ${pids[*]}"
