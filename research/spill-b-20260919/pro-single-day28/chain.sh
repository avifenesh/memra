#!/usr/bin/env bash
# Day 28 target-card chain (one RTX PRO 6000 Blackwell, 27B, MEMRA_CTX unset): build the lane tip, then the shape-walk
# cell through the collector on the canonical /tmp/memra-gpu.lock; bounded wait when another lane holds it, never a
# signal to anything.
set -uo pipefail
R=/root/spill-receipts/b-day28; mkdir -p $R
cd /root/wt-b || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> $R/chain.log; }
log "chain start HEAD=$(git rev-parse HEAD)"
cargo build --release -p memra-server --offline > $R/build.log 2>&1; rc=$?; echo "exit=$rc" >> $R/build.log
git rev-parse HEAD > $R/source.txt
[ $rc = 0 ] || { log "build failed rc=$rc"; exit 1; }
sha256sum target/release/memra-server > $R/binary.sha256
run_cell() {
  local name=$1; shift
  local deadline=$((SECONDS + 3600))
  while ! flock -n /tmp/memra-gpu.lock true; do
    [ $SECONDS -ge $deadline ] && { log "$name: lock still held after 3600 s; not run"; return 3; }
    sleep 30
  done
  log "$name start"
  rm -rf $R/cell-$name
  python3 tools/tier-battery.py --rig pro-single --timeout 3600 --out $R/cell-$name --execute \
    env WT=/root/wt-b RIGDIR=$R/cells MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf MODEL_KEY=q38 LOCK=none NO_SCOPE=1 PORT=18528 CTX=unset "$@" \
    bash research/spill-b-20260919/run-day28-cell.sh $name > $R/collector-$name.log 2>&1
  log "$name exit=$? (cell exit $(cat $R/cells/$name.exit 2>/dev/null))"
  sleep 5
}
run_cell before
log "chain done"
