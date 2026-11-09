#!/usr/bin/env bash
# memra#1093 follow-up: warm == cold pair cells across the spec K table's cached-long rung, RTX 5090.
# usage: k_pairs.sh BINARY OUT EXPECT [CASE-NAME...]
#   EXPECT: kdiff (the base: a warm turn runs another K than its cold twin) or ksame (the fix), for
#   the sampled cases; the greedy cases expect kdiff on both binaries.
#   A YIELD file in OUT, or a gpulock waiter whose reason says "landing", stops between cases.
set -u
B=$1 O=$2 X=$3
shift 3
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O"
ALL=(
  # greedy keeps the cached-long rung on both binaries (K-invariant): its K differs, its bytes match
  "chat-long2-open=$B|chat-long2|open,open,open|_EXPECT=kdiff"
  "chat-long2-open-t1=$B|chat-long2|open,open,open|_TEMP=1.0 _SEED=1234 _EXPECT=$X"
  "chat-long2-open-t1-pen=$B|chat-long2|open,open,open|_TEMP=1.0 _SEED=1234 _FREQ=0.3 _PRES=0.2 _EXPECT=$X"
  "raw-long2-open=$B|raw-long2|open,open,open|_EXPECT=kdiff"
  "raw-long2-open-t1=$B|raw-long2|open,open,open|_TEMP=1.0 _SEED=1234 _EXPECT=$X"
  "raw-long2-nopc-t1=$B|raw-long2|open,open,open|MEMRA_PREFIX_CACHE_MB=0 _USER=k-pair-1 _TEMP=1.0 _SEED=1234 _EXPECT=$X"
)
for c in "${ALL[@]}"; do
  n=${c%%=*}
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$n"; then continue; fi
  if [ -e "$O/YIELD" ]; then echo "yield before $n $(date -Is)"; break; fi
  if gpulock status 2>/dev/null | grep -qi "waiting:.*landing"; then
    echo "landing waiter queued; yield before $n $(date -Is)"; touch "$O/YIELDED-landing"; break
  fi
  echo "case $n start $(date -Is)"
  CELL_PORT=18193 python3 "$H/k_pair_cells.py" "$O/pairs" "$c" >> "$O/pairs.log" 2>&1
  echo "case $n exit $? $(date -Is)"
done
echo PAIRS_DONE
