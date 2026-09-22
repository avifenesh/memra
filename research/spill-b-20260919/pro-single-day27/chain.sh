#!/usr/bin/env bash
# Day 27 target-card chain (one RTX PRO 6000 Blackwell, MEMRA_CTX unset): build, the after cell, then the park cell
# (order 1: default then compact; order 2: compact then default), every run through the collector on the canonical
# /tmp/memra-gpu.lock; bounded wait when another lane holds it, never a signal to anything.
set -uo pipefail
R=/root/spill-receipts/b-day27
cd /root/wt-b || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> $R/chain.log; }
log "chain start HEAD=$(git rev-parse HEAD)"
cargo build --release -p memra-server --offline > $R/build.log 2>&1; rc=$?; echo "exit=$rc" >> $R/build.log
git rev-parse HEAD > $R/source.txt
[ $rc = 0 ] || { log "build failed rc=$rc"; exit 1; }
sha256sum target/release/memra-server > $R/binary.sha256
run_cell() {
  local name=$1 order=$2; shift 2
  local deadline=$((SECONDS + 1800))
  while ! flock -n /tmp/memra-gpu.lock true; do
    [ $SECONDS -ge $deadline ] && { log "$name: lock still held after 1800 s; not run"; return 3; }
    sleep 30
  done
  log "$name start"
  rm -rf $R/cell-$name
  python3 tools/tier-battery.py --rig pro-single --timeout 3600 --out $R/cell-$name --execute \
    env WT=/root/wt-b RIGDIR=$R/cells MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf MODEL_KEY=q38 LOCK=none NO_SCOPE=1 PORT=18526 "$@" \
    bash research/spill-b-20260919/run-day26-cell.sh $name $order > $R/collector-$name.log 2>&1
  log "$name exit=$? (cell exit $(cat $R/cells/$name.exit 2>/dev/null))"
  sleep 5
}
run_cell after-ab AB
run_cell park-o1-default AB
run_cell park-o1-compact AB MEMRA_KV_PARK_COMPACT=1
run_cell park-o2-compact AB MEMRA_KV_PARK_COMPACT=1
run_cell park-o2-default AB
log "chain done"
