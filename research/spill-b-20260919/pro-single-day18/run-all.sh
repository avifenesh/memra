#!/usr/bin/env bash
# Day 18 target-card chain on the fix binary (bins/fix, built at the lane tip by build-arm.sh): the twin
# gate on the day-16 shape, the restore-identity gate on the day-17 five points, serve-smoke, the
# cache-metering arm, the prefix-evict-reclaim gate, lane A's kv-host tenant-reclaim gate (fix arm), then
# the collector's --validate over every cell. Every GPU command goes through the collector on the
# canonical /tmp/memra-gpu.lock; gates that take their own lock inherit the collector's FD
# (--external-lock @COLLECTOR_LOCK_FD@); serve-smoke and cache-meter run with the collector holding
# the lock (day-14 shape). Bounded lock retries, 90 s apart, never killing a holder (lane C may run).
set -uo pipefail
R=/root/spill-receipts/b-day18
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
BIN=$R/bins/fix/memra-server
cd /root/wt-b
git rev-parse HEAD > $R/gate-source.txt
sha256sum $BIN > $R/gate-binary.sha256
busy() { grep -q "Resource temporarily unavailable\|canonical rig lock\|BlockingIOError\|lock busy" "$1" && [ ! -f "$2/command.log" ]; }
cell() { # $1 name $2 timeout $3 external(1|0) $4... command
  local name=$1 tmo=$2 ext=$3; shift 3
  for attempt in $(seq 0 24); do
    out=$R/$name; [ $attempt -gt 0 ] && out=$R/$name-retry$attempt
    nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-$name.csv
    if [ "$ext" = 1 ]; then
      python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" --external-lock --execute "${@//@OUT@/$out}" > "$out-driver.log" 2>&1
    else
      python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" --execute "${@//@OUT@/$out}" > "$out-driver.log" 2>&1
    fi
    rc=$?; echo $rc > "$out.exit"
    if busy "$out-driver.log" "$out"; then echo "$name attempt $attempt: lock busy at $(date -u +%T)" >> $R/retries.log; sleep 90; continue; fi
    echo "$name rc=$rc attempt=$attempt out=$(basename $out)" >> $R/chain.log; break
  done
}
cell gate-fix-day16-shape 2400 1 python3 tools/prefix-newest-turn-fits-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
  --model $MODEL --bin $BIN --out @OUT@/cell --gpu-lock /tmp/memra-gpu.lock \
  --budget-mib 1024 --cohort-tokens 1250,1350,1450,1550 --turns 12 --start-tokens 11000 --grow-tokens 150
cell restore-fix 1500 1 python3 tools/prefix-restore-identity-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
  --model $MODEL --bin $BIN --out @OUT@/cell --gpu-lock /tmp/memra-gpu.lock
cell serve-smoke 3000 0 bash tools/serve-smoke.sh $MODEL /nonexistent-draft
sha256sum target/release/memra-server > $R/serve-smoke-binary.sha256
cp /tmp/serve-smoke.log $R/serve-smoke-server.log 2>/dev/null; rm -f /tmp/serve-smoke.log
cell cache-meter 1500 0 bash $R/cache-meter-cell.sh
cell evict-reclaim 2400 1 python3 tools/prefix-evict-reclaim-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
  --model $MODEL --bin $BIN --out @OUT@/cell --gpu-lock /tmp/memra-gpu.lock
cell kv-host-fix 2400 1 env MEMRA_GPU_LOCK=/tmp/memra-gpu.lock bash tools/kv-host-tenant-reclaim-gate.sh --external-lock @COLLECTOR_LOCK_FD@ fix $MODEL $BIN @OUT@/cell
for c in $(ls -d $R/*/ | xargs -n1 basename | grep -v "^build-\|^bins$"); do
  [ -f $R/$c/command.log ] && { python3 tools/tier-battery.py --validate $R/$c > $R/validate-$c.log 2>&1; echo "validate-$c rc=$?" >> $R/chain.log; }
done
echo "done $(date -u +%FT%TZ)" >> $R/chain.log
