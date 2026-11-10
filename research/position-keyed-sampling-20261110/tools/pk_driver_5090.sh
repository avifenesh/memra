#!/usr/bin/env bash
# memra#1154 prototype on the RTX 5090: one boot per gpulock lease.
#   perf  12 boots base cand cand base x3 (pk_ab.py MODE=perf), the first after the card cooled
#   inv   per binary and per pinned K in 0 1 2 3: part x, then part y with x's turn-1 answer
# Stops before a lease when a gpulock waiter says "landing" or OUT/YIELD exists.
# usage: pk_driver_5090.sh BASE_BIN CAND_BIN OUT [perf|inv|all]
set -u
BASE=$1 CAND=$2 O=$3 WHAT=${4:-all}
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O/perf" "$O/inv"
lease() {  # lease NAME -- CMD...
  local name=$1; shift 2
  if [ -e "$O/YIELD" ]; then echo "yield before $name $(date -Is)"; return 9; fi
  if gpulock status 2>/dev/null | grep -qi "waiting:.*landing"; then
    echo "landing waiter queued; yield before $name $(date -Is)"; touch "$O/YIELDED-landing"; return 9
  fi
  echo "$name start $(date -Is)"
  gpulock run --ttl 15m --wait 300m --for "memra #1154 position-keyed sampling A/B, $name" -- "$@"
  echo "$name exit $? $(date -Is)"
}
bin_of() { [ "$1" = base ] && echo "$BASE" || echo "$CAND"; }
if [ "$WHAT" = perf ] || [ "$WHAT" = all ]; then
  waitfor --until 'test "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 62' --timeout 10m --every 20s --quiet
  echo "card at $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C before the first boot"
  n=0
  for arm in base cand cand base base cand cand base base cand cand base; do
    n=$((n + 1))
    lease "perf boot $n ($arm)" -- env MODE=perf CELL_PORT=18194 BOOT_START=$n python3 "$H/pk_ab.py" "$O/perf" "$(bin_of $arm)" "$arm" >> "$O/perf/boots.log" 2>&1 || break
  done
fi
if [ "$WHAT" = inv ] || [ "$WHAT" = all ]; then
  n=0
  for arm in cand base; do
    for k in 0 1 2 3; do
      n=$((n + 1))
      lease "inv boot $n ($arm K=$k x)" -- env MODE=inv PART=x SPEC_K=$k CELL_PORT=18194 BOOT_START=$n python3 "$H/pk_ab.py" "$O/inv" "$(bin_of $arm)" "$arm" >> "$O/inv/boots.log" 2>&1 || break 2
      n=$((n + 1))
      lease "inv boot $n ($arm K=$k y)" -- env MODE=inv PART=y SPEC_K=$k E1_FILE="$O/inv/texts/E1-$arm-k$k.txt" CELL_PORT=18194 BOOT_START=$n python3 "$H/pk_ab.py" "$O/inv" "$(bin_of $arm)" "$arm" >> "$O/inv/boots.log" 2>&1 || break 2
    done
  done
fi
echo DRIVER_DONE
