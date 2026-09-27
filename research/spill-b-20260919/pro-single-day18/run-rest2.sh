#!/usr/bin/env bash
# Day 18 target-card follow-up: the chain's restore cell ran the gate's mistaken default (the chain's turn
# 12, 12,650 ids; kept as restore-fix). The day-17 shape is turn 10 (12,350 ids): wait for the chain, then
# the restore gate at --turns 10 on the fix, then validate.
set -uo pipefail
R=/root/spill-receipts/b-day18
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
BIN=$R/bins/fix/memra-server
cd /root/wt-b
for _ in $(seq 1 480); do grep -q "^done" $R/chain.log 2>/dev/null && break; sleep 15; done
busy() { grep -q "Resource temporarily unavailable\|canonical rig lock\|BlockingIOError\|lock busy" "$1" && [ ! -f "$2/command.log" ]; }
for attempt in $(seq 0 24); do
  out=$R/restore-fix-t10; [ $attempt -gt 0 ] && out=$R/restore-fix-t10-retry$attempt
  nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-restore-fix-t10.csv
  python3 tools/tier-battery.py --rig pro-single --timeout 1500 --out "$out" --external-lock --execute python3 tools/prefix-restore-identity-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
    --model $MODEL --bin $BIN --out "$out/cell" --gpu-lock /tmp/memra-gpu.lock --turns 10 > "$out-driver.log" 2>&1
  rc=$?; echo $rc > "$out.exit"
  if busy "$out-driver.log" "$out"; then echo "restore-fix-t10 attempt $attempt: lock busy at $(date -u +%T)" >> $R/retries.log; sleep 90; continue; fi
  echo "restore-fix-t10 rc=$rc attempt=$attempt out=$(basename $out)" >> $R/chain.log; break
done
[ -f $R/restore-fix-t10/command.log ] && { python3 tools/tier-battery.py --validate $R/restore-fix-t10 > $R/validate-restore-fix-t10.log 2>&1; echo "validate-restore-fix-t10 rc=$?" >> $R/chain.log; }
echo "rest2 done $(date -u +%FT%TZ)" >> $R/chain.log
