#!/usr/bin/env bash
# Day 15 landed-binary battery, second sitting. The first sitting's three GPU cells were REFUSED by the
# collector (`REFUSED: [Errno 11] Resource temporarily unavailable`: another lane's collector cell held
# the canonical lock); they are kept under refused-lockbusy/. Same binary (bins/landed, built at the
# landing ref, build-landed/). Bounded lock retries, 90 s apart, never killing a holder; every attempt
# leaves its own driver log; the collector creates --out itself and refuses one that exists.
set -uo pipefail
R=/root/spill-receipts/b-day15
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-b
git rev-parse HEAD > $R/gate-source-landed.txt
busy() { grep -q "Resource temporarily unavailable\|canonical rig lock\|BlockingIOError\|lock busy" "$1" && [ ! -f "$2/command.log" ]; }
# 1. the day-14 twin gate on the landed binary (its own inherited-lock FD)
for attempt in $(seq 0 24); do
  out=$R/gate-landed; [ $attempt -gt 0 ] && out=$R/gate-landed-retry$attempt
  python3 tools/tier-battery.py --rig pro-single --timeout 1800 --out "$out" --external-lock \
    --execute python3 tools/prefix-newest-turn-fits-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
      --model /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf --bin $R/bins/landed/memra-server \
      --out "$out/cell" --budget-mib 1024 --cohort-tokens 2800,3000,3200 \
      --turns 8 --start-tokens 9200 --grow-tokens 300 > "$out-driver.log" 2>&1
  rc=$?; echo $rc > "$out.exit"
  if busy "$out-driver.log" "$out"; then echo "gate-landed attempt $attempt: lock busy at $(date -u +%T)" >> $R/landed2-retries.log; sleep 90; continue; fi
  echo "gate-landed rc=$rc attempt=$attempt out=$(basename $out)" >> $R/landed2.log; break
done
# 2. serve-smoke (the collector holds the lock; no FD placeholder)
for attempt in $(seq 0 24); do
  out=$R/serve-smoke; [ $attempt -gt 0 ] && out=$R/serve-smoke-retry$attempt
  python3 tools/tier-battery.py --rig pro-single --timeout 3000 --out "$out" --execute bash tools/serve-smoke.sh /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf /nonexistent-draft > "$out-driver.log" 2>&1
  rc=$?; echo $rc > "$out.exit"
  if busy "$out-driver.log" "$out"; then echo "serve-smoke attempt $attempt: lock busy at $(date -u +%T)" >> $R/landed2-retries.log; sleep 90; continue; fi
  sha256sum target/release/memra-server > $R/serve-smoke-binary.sha256
  cp /tmp/serve-smoke.log $R/serve-smoke-server.log 2>/dev/null; rm -f /tmp/serve-smoke.log
  echo "serve-smoke rc=$rc attempt=$attempt out=$(basename $out)" >> $R/landed2.log; break
done
# 3. cache-meter on the landed binary
for attempt in $(seq 0 24); do
  out=$R/cache-meter; [ $attempt -gt 0 ] && out=$R/cache-meter-retry$attempt
  python3 tools/tier-battery.py --rig pro-single --timeout 1500 --out "$out" --execute bash $R/cache-meter-cell.sh > "$out-driver.log" 2>&1
  rc=$?; echo $rc > "$out.exit"
  if busy "$out-driver.log" "$out"; then echo "cache-meter attempt $attempt: lock busy at $(date -u +%T)" >> $R/landed2-retries.log; sleep 90; continue; fi
  echo "cache-meter rc=$rc attempt=$attempt out=$(basename $out)" >> $R/landed2.log; break
done
for c in ab-smoke ab-full $(ls -d $R/gate-landed* $R/serve-smoke* $R/cache-meter* 2>/dev/null | xargs -n1 basename | grep -v "\.\(exit\|log\|sha256\)$"); do
  [ -d $R/$c ] && [ -f $R/$c/command.log ] && { python3 tools/tier-battery.py --validate $R/$c > $R/validate-$c.log 2>&1; echo "validate-$c rc=$?" >> $R/landed2.log; }
done
echo done >> $R/landed2.log
