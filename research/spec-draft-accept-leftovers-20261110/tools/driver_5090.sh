#!/usr/bin/env bash
# memra#1169 on the RTX 5090: exactness first, in two leases (exact_5090.sh gates, then eager),
# then the perf cell, 12 boots before after after before x3, one boot per lease, the first
# after the card cooled to 62 C. Waits while a "landing" lease holds the card or waits for it. A failed exactness
# lease stops the driver.
# usage: driver_5090.sh STAGE...   (STAGE = gates | eager | perf; PERF_SKIP=n resumes after boot n)
set -u
H=$HOME/.cache/memra-1160/run
O=$H/rtx5090
B=$HOME/.cache/memra-1160/bin
mkdir -p "$O/perf"
lease() { local ttl=$1 name=$2; shift 3
  [ -e "$O/YIELD" ] && { echo "yield before $name $(date -Is)"; return 9; }
  waitfor --until "bash $H/no_landing.sh" --timeout 240m --every 30s --quiet
  echo "$name start $(date -Is)"
  gpulock run --ttl "$ttl" --wait 300m --for "memra draft-side leftovers: $name" -- "$@"
  local rc=$?
  echo "$name exit $rc $(date -Is)"
  return $rc
}
stop() { echo "STOP: $1 failed $(date -Is)"; exit 1; }
for stage in "$@"; do
  case $stage in
    gates)
      lease 30m "seeded-invariance gate, after then before (short boots, pass/fail)" -- bash "$H/exact_5090.sh" gates || stop "gates" ;;
    eager)
      lease 35m "seeded-invariance gate on the eager draft chain, after then before, and the pooled-draft probe (pass/fail)" -- bash "$H/exact_5090.sh" eager || stop "eager gates or probe" ;;
    perf)
      waitfor --until 'test "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 62' --timeout 10m --every 20s --quiet
      echo "card at $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C before the first perf boot"
      n=0
      for arm in before after after before before after after before before after after before; do
        n=$((n + 1))
        [ "$n" -le "${PERF_SKIP:-0}" ] && continue
        lease 15m "perf boot $n ($arm, one boot per lease)" -- env MODE=perf CELL_PORT=18194 BOOT_START=$n python3 "$H/ab_cell.py" "$O/perf" "$B/memra-server-$arm" "$arm" >> "$O/perf/boots.log" 2>&1 || stop "perf boot $n"
      done ;;
  esac
done
echo DRIVER_DONE
