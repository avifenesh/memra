#!/usr/bin/env bash
# memra#1093 item 1: warm == cold pair cells for the MTP spec pool on the RTX 5090.
# usage: pairs.sh BINARY OUT EXPECT [CASE-NAME...]
#   EXPECT: offgrid (the base binary: the case must reach an off-lineage resume) or grid (the fix).
#   A YIELD file in OUT, or a gpulock waiter whose reason says "landing", stops between cases.
set -u
B=$1 O=$2 X=$3
shift 3
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O"
ALL=(
  # shapes that reach an off-lineage resume on main (EXPECT decides what the warm log must show)
  "chat-short-open=$B|chat-short|open,open,open|_EXPECT=$X"
  "chat-short-open-t1=$B|chat-short|open,open,open|_TEMP=1.0 _SEED=1234 _EXPECT=$X"
  "chat-short-open-t1-pen=$B|chat-short|open,open,open|_TEMP=1.0 _SEED=1234 _FREQ=0.3 _PRES=0.2 _EXPECT=$X"
  "chat-short-256=$B|chat-short|256,256,256|_EXPECT=$X"
  "chat-short-256-t1=$B|chat-short|256,256,256|_TEMP=1.0 _SEED=1234 _EXPECT=$X"
  "chat-mid-open=$B|chat-mid|open,open,open|_EXPECT=$X"
  "raw-short-open=$B|raw-short|open,open,open|_EXPECT=$X"
  "raw-long-nopc=$B|raw-long|open,open,open|MEMRA_PREFIX_CACHE_MB=0 _USER=pair-1 _EXPECT=$X"
  "raw-long-nopc-t1=$B|raw-long|open,open,open|MEMRA_PREFIX_CACHE_MB=0 _USER=pair-1 _TEMP=1.0 _SEED=1234 _EXPECT=$X"
  # an exact extension of a session whose checkpoint is on the grid: the fix rewinds the exact tier
  "chat-grid-open=$B|chat-grid|open,open,open|_EXPECT=$X"
  "chat-grid-open-t1=$B|chat-grid|open,open,open|_TEMP=1.0 _SEED=1234 _EXPECT=$X"
  "chat-grid-open-t1-pen=$B|chat-grid|open,open,open|_TEMP=1.0 _SEED=1234 _FREQ=0.3 _PRES=0.2 _EXPECT=$X"
  "raw-grid-open=$B|raw-grid|open,open,open|_EXPECT=$X"
  "chat-exact-open=$B|chat-exact|open,open,open|_EXPECT=$X"
  "chat-exact-open-t1=$B|chat-exact|open,open,open|_TEMP=1.0 _SEED=1234 _EXPECT=$X"
  "chat-exact-open-t1-pen=$B|chat-exact|open,open,open|_TEMP=1.0 _SEED=1234 _FREQ=0.3 _PRES=0.2 _EXPECT=$X"
  # on-grid controls: main already rewinds these on the grid
  "chat-story-open=$B|chat-story|open,open,open|_EXPECT=grid"
  "chat-story-open-t1=$B|chat-story|open,open,open|_TEMP=1.0 _SEED=1234 _EXPECT=grid"
  "raw-story-open-t1=$B|raw-story|open,open,open|_TEMP=1.0 _SEED=1234"
  "chat-think-1024=$B|chat-think|1024,1024,1024|_EXPECT=grid"
  "chat-think-1024-t1=$B|chat-think|1024,1024,1024|_TEMP=1.0 _SEED=1234 _EXPECT=grid"
  "chat-think-1024-t1-pen=$B|chat-think|1024,1024,1024|_TEMP=1.0 _SEED=1234 _FREQ=0.3 _PRES=0.2 _EXPECT=grid"
)
for c in "${ALL[@]}"; do
  n=${c%%=*}
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$n"; then continue; fi
  if [ -e "$O/YIELD" ]; then echo "yield before $n $(date -Is)"; break; fi
  if gpulock status 2>/dev/null | grep -qi "waiting:.*landing"; then
    echo "landing waiter queued; yield before $n $(date -Is)"; touch "$O/YIELDED-landing"; break
  fi
  echo "case $n start $(date -Is)"
  CELL_PORT=18193 python3 "$H/spec_pool_pair_cells.py" "$O/pairs" "$c" >> "$O/pairs.log" 2>&1
  echo "case $n exit $? $(date -Is)"
done
echo PAIRS_DONE
