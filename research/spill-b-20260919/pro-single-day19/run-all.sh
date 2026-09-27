#!/usr/bin/env bash
# Day 19 target-card chain: build the fix at <ref> (fetched from the pushed lane branch), then on the
# fix: the twin gate (day-16 shape), the restore gate (turn 10), serve-smoke, cache-meter, lane A's
# kv-host tenant-reclaim fix arm, the evict-reclaim gate; then the evict-reclaim gate on the base
# binary built here on day 18 from origin/main 1b354be59 (b-day18/bins/base, sha recorded); then
# --validate over every cell. Collector on /tmp/memra-gpu.lock, inherited where the gate takes a
# lock; bounded retries 90 s apart when another lane holds it (lane C runs beside this today).
# usage: run-all.sh <fix-ref>
set -uo pipefail
ref=${1:?fix ref}
R=/root/spill-receipts/b-day19
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p $R
bash $R/build-arm.sh fix "$ref"; echo "build-fix rc=$? exit=$(cat $R/build-fix/exit)" >> $R/chain.log
[ "$(cat $R/build-fix/exit)" = 0 ] || { echo "build failed; chain stops" >> $R/chain.log; exit 1; }
BIN=$R/bins/fix/memra-server
BASE=/root/spill-receipts/b-day18/bins/base/memra-server
sha256sum $BASE > $R/base-binary.sha256; cp /root/spill-receipts/b-day18/build-base/source.txt $R/base-source.txt
cd /root/wt-b
git rev-parse HEAD > $R/gate-source.txt
sha256sum $BIN > $R/gate-binary.sha256
busy() { grep -q "Resource temporarily unavailable\|canonical rig lock\|BlockingIOError\|lock busy" "$1" && [ ! -f "$2/command.log" ]; }
cell() { # $1 name $2 timeout $3 external(1|0) $4... command (@OUT@ = the cell dir)
  local name=$1 tmo=$2 ext=$3; shift 3
  for attempt in $(seq 0 30); do
    out=$R/$name; [ $attempt -gt 0 ] && out=$R/$name-retry$attempt
    nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-$name.csv
    if [ "$ext" = 1 ]; then
      python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" --external-lock --execute "${@//@OUT@/$out}" > "$out-driver.log" 2>&1
    else
      python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" --execute "${@//@OUT@/$out}" > "$out-driver.log" 2>&1
    fi
    rc=$?; echo $rc > "$out.exit"
    if busy "$out-driver.log" "$out"; then echo "$name attempt $attempt: lock busy at $(date -u +%T)" >> $R/retries.log; sleep 90; continue; fi
    echo "$name rc=$rc attempt=$attempt out=$(basename $out)" >> $R/chain.log
    [ -f "$out/command.log" ] && { python3 tools/tier-battery.py --validate "$out" > "$R/validate-$name.log" 2>&1; echo "validate-$name rc=$?" >> $R/chain.log; }
    break
  done
}
cell gate-fix-day16-shape 2400 1 python3 tools/prefix-newest-turn-fits-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
  --model $MODEL --bin $BIN --out @OUT@/cell --gpu-lock /tmp/memra-gpu.lock \
  --budget-mib 1024 --cohort-tokens 1250,1350,1450,1550 --turns 12 --start-tokens 11000 --grow-tokens 150
cell restore-fix-t10 1500 1 python3 tools/prefix-restore-identity-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
  --model $MODEL --bin $BIN --out @OUT@/cell --gpu-lock /tmp/memra-gpu.lock --turns 10
cell serve-smoke 3000 0 bash tools/serve-smoke.sh $MODEL /nonexistent-draft
sha256sum target/release/memra-server > $R/serve-smoke-binary.sha256
cp /tmp/serve-smoke.log $R/serve-smoke-server.log 2>/dev/null; rm -f /tmp/serve-smoke.log
cell cache-meter 1500 0 bash $R/cache-meter-cell.sh
cell kv-host-fix 2400 1 env MEMRA_GPU_LOCK=/tmp/memra-gpu.lock bash tools/kv-host-tenant-reclaim-gate.sh --external-lock @COLLECTOR_LOCK_FD@ fix $MODEL $BIN @OUT@/cell
cell evict-reclaim-fix 2400 1 python3 tools/prefix-evict-reclaim-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
  --model $MODEL --bin $BIN --out @OUT@/cell --gpu-lock /tmp/memra-gpu.lock
cell evict-reclaim-base 2400 1 python3 tools/prefix-evict-reclaim-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
  --model $MODEL --bin $BASE --out @OUT@/cell --gpu-lock /tmp/memra-gpu.lock
echo "done $(date -u +%FT%TZ)" >> $R/chain.log
