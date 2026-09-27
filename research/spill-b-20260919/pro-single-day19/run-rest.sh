#!/usr/bin/env bash
# Day 19 target-card follow-up: wait for the chain, check out the gate tip <ref> (fetched from the
# bundle), and rerun lane A's kv-host tenant-reclaim fix arm on the gate that reads the aligned capture
# (its first run failed on the two promote equalities that still read the whole leader prompt).
# usage: run-rest.sh <gate-ref>
set -uo pipefail
ref=${1:?gate ref}
R=/root/spill-receipts/b-day19
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
BIN=$R/bins/fix/memra-server
cd /root/wt-b
for _ in $(seq 1 600); do grep -q "^done" $R/chain.log 2>/dev/null && break; sleep 15; done
git checkout -q --detach "$ref" || { echo "gate tip checkout failed" >> $R/chain.log; exit 1; }
git rev-parse HEAD > $R/gate-source-rest.txt
busy() { grep -q "Resource temporarily unavailable\|canonical rig lock\|BlockingIOError\|lock busy" "$1" && [ ! -f "$2/command.log" ]; }
for attempt in $(seq 0 30); do
  out=$R/kv-host-fix-2; [ $attempt -gt 0 ] && out=$R/kv-host-fix-2-retry$attempt
  nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw --format=csv,noheader > $R/card-before-kv-host-fix-2.csv
  python3 tools/tier-battery.py --rig pro-single --timeout 2400 --out "$out" --external-lock --execute env MEMRA_GPU_LOCK=/tmp/memra-gpu.lock bash tools/kv-host-tenant-reclaim-gate.sh --external-lock @COLLECTOR_LOCK_FD@ fix $MODEL $BIN "$out/cell" > "$out-driver.log" 2>&1
  rc=$?; echo $rc > "$out.exit"
  if busy "$out-driver.log" "$out"; then echo "kv-host-fix-2 attempt $attempt: lock busy at $(date -u +%T)" >> $R/retries.log; sleep 90; continue; fi
  echo "kv-host-fix-2 rc=$rc attempt=$attempt out=$(basename $out)" >> $R/chain.log
  [ -f "$out/command.log" ] && { python3 tools/tier-battery.py --validate "$out" > $R/validate-kv-host-fix-2.log 2>&1; echo "validate-kv-host-fix-2 rc=$?" >> $R/chain.log; }
  break
done
echo "rest done $(date -u +%FT%TZ)" >> $R/chain.log
