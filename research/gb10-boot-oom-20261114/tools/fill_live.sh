#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): fill the page cache with four readers
# until MemFree is under LIMIT_MB (memra#1257's fill_cache2.sh, on main), then keep READERS readers
# going over the model files, largest first, until STOP_FILE exists or KEEP_S seconds pass, so the
# page cache keeps filling and the kernel keeps reclaiming while boots run. This is memra#1307's
# live_fill.sh with the readers' pids written to PIDS_FILE, so the caller stops exactly those.
# MemFree, MemAvailable and Cached at 1 Hz go to MEMLOG. Reads only; changes no setting.
# The fill gives up after FILL_CAP_S seconds (300): with several readers the kernel can hold
# MemFree above the limit for minutes.
# usage: [FILL_CAP_S=n] fill_live.sh LIMIT_MB READERS KEEP_S STOP_FILE MEMLOG PIDS_FILE
set -u
LIM=$1; NR=$2; KEEP=$3; STOP=$4; MEMLOG=$5; PIDS=$6
H=$(cd "$(dirname "$0")" && pwd)
mi() { grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | awk '{printf "%s %d ", $1, $2/1024}'; }
(
  while [ ! -e "$STOP" ]; do echo "$(date +%s.%N | cut -c1-14) $(mi)"; sleep 1; done
) >> "$MEMLOG" 2>/dev/null < /dev/null &
echo $! >> "$PIDS"
echo "$(date -Is) fill_live: before $(mi)"
bash "$H/../../admit-headroom-unified-20261113/tools/fill_cache2.sh" "$LIM" "${FILL_CAP_S:-300}" 4 | sed 's/^/  /'
L=$(mktemp)
ls -S "$HOME"/models/*.gguf "$HOME"/models/*/*.gguf "$HOME"/models/*/*/*.gguf "$HOME"/models/*/*.safetensors \
  "$HOME"/models/*/*/*.safetensors 2>/dev/null > "$L"
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
  echo $! >> "$PIDS"
done
echo "$(date -Is) fill_live: $NR readers running until $STOP (at most ${KEEP}s)"
