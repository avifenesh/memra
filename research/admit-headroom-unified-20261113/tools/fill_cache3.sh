#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1257): fill_cache2.sh's fill (four readers,
# MemFree under LIMIT_MB, default 3000), then ONE reader keeps reading model files in the
# background while the cell's memra-server lives (at most KEEP_S seconds, default 900), so the page
# cache stays full and MemFree stays near its low point through the cold request's prime, not
# only at the first request (cell 11-B: the rung's raw reading was 1,929 MiB before g1k's prime and
# back over the floor by g260k's). Reads only; the kernel reclaims these pages on demand.
# usage: fill_cache3.sh [LIMIT_MB [CAP_S [KEEP_S]]]
LIM=${1:-3000}; CAP=${2:-300}; KEEP=${3:-900}
H=$(cd "$(dirname "$0")" && pwd)
bash "$H/fill_cache2.sh" "$LIM" "$CAP" 4
spid=$(pgrep -x memra-server | head -n 1)
[ -n "$spid" ] || { echo "$(date -Is) fill3: no memra-server running; no background reader"; exit 0; }
(
  end=$((SECONDS + KEEP))
  while kill -0 "$spid" 2>/dev/null && [ $SECONDS -lt $end ]; do
    for f in $(ls -S $HOME/models/*/*.safetensors $HOME/models/*/*/*.safetensors $HOME/models/*/*.gguf 2>/dev/null | shuf); do
      kill -0 "$spid" 2>/dev/null && [ $SECONDS -lt $end ] || break
      timeout 30 cat "$f" > /dev/null
    done
  done
  echo "$(date -Is) fill3: background reader stopped" >> /dev/null
) > /dev/null 2>&1 < /dev/null &
echo "$(date -Is) fill3: background reader $! follows memra-server $spid (at most ${KEEP}s)"
# MemFree at 1 Hz while the server lives, matched to the cell by the server's pid (the raw driver
# reading on GB10 tracks MemFree; serve.sh logs MemAvailable only).
M=$HOME/.cache/memra-1257/memfree-$spid.log
(
  while kill -0 "$spid" 2>/dev/null; do
    echo "$(date +%s.%N | cut -c1-14) $(awk '/^MemFree/{print $2}' /proc/meminfo)"
    sleep 1
  done
) > "$M" 2>/dev/null < /dev/null &
