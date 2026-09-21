#!/usr/bin/env bash
# Day 18 target-card follow-up 2: attribution arms. (1) The evict-reclaim gate read V3=FAIL on the fix
# (pool_retained_bytes=148013056); the same gate on a base binary built here from origin/main 1b354be59
# says whether the fix moved it. (2) Lane A's kv-host gate fix arm again with the gate as committed at
# the lane tip (its promote equalities read r1/r6 prompt_tokens: the entries are spec publications).
set -uo pipefail
R=/root/spill-receipts/b-day18
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
cd /root/wt-b
for _ in $(seq 1 480); do grep -q "^rest2 done" $R/chain.log 2>/dev/null && break; sleep 15; done
bash $R/build-arm.sh base 1b354be594acc92d9dc28d6421a93538cecde202; echo "build-base rc=$? exit=$(cat $R/build-base/exit)" >> $R/chain.log
git fetch -q origin "+refs/heads/lane/spill-b-20260919:refs/remotes/origin/lane/spill-b-20260919"
git checkout -q --detach 66c0aea89e3b926c4b6489c12565752108b137cc || { echo "gate tip checkout failed" >> $R/chain.log; exit 1; }
git rev-parse HEAD > $R/gate-source-rest3.txt
busy() { grep -q "Resource temporarily unavailable\|canonical rig lock\|BlockingIOError\|lock busy" "$1" && [ ! -f "$2/command.log" ]; }
cell() { local name=$1 tmo=$2; shift 2
  for attempt in $(seq 0 24); do
    out=$R/$name; [ $attempt -gt 0 ] && out=$R/$name-retry$attempt
    nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-$name.csv
    python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" --external-lock --execute "${@//@OUT@/$out}" > "$out-driver.log" 2>&1
    rc=$?; echo $rc > "$out.exit"
    if busy "$out-driver.log" "$out"; then echo "$name attempt $attempt: lock busy at $(date -u +%T)" >> $R/retries.log; sleep 90; continue; fi
    echo "$name rc=$rc attempt=$attempt out=$(basename $out)" >> $R/chain.log; break
  done
  [ -f $R/$name/command.log ] && { python3 tools/tier-battery.py --validate $R/$name > $R/validate-$name.log 2>&1; echo "validate-$name rc=$?" >> $R/chain.log; }
}
cell evict-reclaim-base 2400 python3 tools/prefix-evict-reclaim-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
  --model $MODEL --bin $R/bins/base/memra-server --out @OUT@/cell --gpu-lock /tmp/memra-gpu.lock
cell kv-host-fix-2 2400 env MEMRA_GPU_LOCK=/tmp/memra-gpu.lock bash tools/kv-host-tenant-reclaim-gate.sh --external-lock @COLLECTOR_LOCK_FD@ fix $MODEL $R/bins/fix/memra-server @OUT@/cell
echo "rest3 done $(date -u +%FT%TZ)" >> $R/chain.log
