#!/usr/bin/env bash
# Both door arms of the capture cell in ONE collector lock hold, both orders: OFF, ON, ON, OFF (each boot's
# harness already interleaves idle/arm in both orders, N=5 per arm per order). usage: stall-both.sh <lockfd>
set -uo pipefail
fd=$1
D=research/spill-a-20260919/pro-single-day21
R=/root/spill-receipts/a-day21
cd /root/wt-a
rc=0
for pass in 1 2; do
  for which in off on; do
    [ $pass = 2 ] && { [ $which = off ] && which=on || which=off; }
    export MEMRA_GATE_PORT=18132
    bash $D/stall-cell.sh "$fd" $which > "$R/stall-restore-$which.pass$pass.log" 2>&1 || rc=1
    # the second pass writes into the same ev dir: move the receipt aside first
    mv "$R/stall-restore-$which/ev" "$R/stall-restore-$which/ev.pass$pass" 2>/dev/null || true
    echo "$(date -u +%FT%TZ) stall-restore-$which pass=$pass rc=$rc"
  done
done
echo stall-both-done
exit $rc
