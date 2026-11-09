#!/usr/bin/env bash
# Rig: lever 1's pooled-cohort cell on the RTX 5090: research/batch-spec-20261101/battery/
# pool_cell.py at POOL_MAX_TOKENS=700 (q9, c = 2, 4, 8 cohorts, greedy and sampled, MEMRA_ADMIT_YIELD=0) on main
# bc235edf53 and on the lever's head, one gpulock lease per boot; every stream must match.
# usage: eos5090_pool.sh HEAD_SHA
set -u
export PATH=/usr/bin:$PATH
D=$HOME/.cache/memra-gb10-r4
B=$D/cells-work/rtx5090/bins/bc235edf538ecc5bc64aa7ec161cfed11e0a3c71-120a
T=$D/cells-work/rtx5090/bins/$1-120a
CD=$D/scout/eos-pool-5090; mkdir -p "$CD"
L=$CD/driver.log
W=$D/wt-eos/research/batch-spec-20261101/battery
P=$D/wt-eos/research/gb10-decode-20261109/tools/pool_cell.py
for arm in B T; do
  bin=$B; [ "$arm" = T ] && bin=$T
  echo "$(date -Is) pool-$arm start" >> "$L"
  gpulock run --ttl 20m --for "memra round 5 lever 1 pooled-cohort cell on the 5090: pool-$arm (one boot per lease)" --wait 90m -- \
    env POOL_MAX_TOKENS=700 /usr/bin/python3 "$P" "$bin/memra-server" "$CD/pool-$arm" 8351 >> "$L" 2>&1
  echo "$(date -Is) pool-$arm rc=$?" >> "$L"
done
/usr/bin/python3 "$W/compare_pool_runs.py" "$CD/pool-B" head="$CD/pool-T" > "$CD/compare-pool.txt" 2>&1
echo "$(date -Is) compare rc=$? all cells done" >> "$L"
