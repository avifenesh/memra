#!/usr/bin/env bash
# WP-B day 31 target-card chain (one RTX PRO 6000 Blackwell, Qwen3.8-27B NVFP4-Q5K MTP, MEMRA_CTX unset): fast-forward
# the lane tip, build, copy the binary once, then one collector hold per order on /tmp/memra-gpu.lock. Before each
# hold: a bounded poll (at most 7200 s) until the lock is free and no compute app is on the card; never a signal to
# anything. Receipts under /root/spill-receipts/b-day31, mirrored to research/spill-b-20260919/pro-single-day31/box/.
set -uo pipefail
R=/root/spill-receipts/b-day31; mkdir -p $R/bins
cd /root/wt-b || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> $R/chain.log; }
log "chain start HEAD=$(git rev-parse HEAD)"
git fetch origin lane/spill-b-20260919 >> $R/fetch.log 2>&1 && git merge --ff-only origin/lane/spill-b-20260919 >> $R/fetch.log 2>&1 || { log "fetch/ff failed"; exit 1; }
log "tree HEAD=$(git rev-parse HEAD)"
git rev-parse HEAD > $R/source.txt
( nice -n 10 cargo build --release -p memra-server --offline || nice -n 10 cargo build --release -p memra-server ) > $R/build.log 2>&1; rc=$?; echo "exit=$rc" >> $R/build.log
[ $rc = 0 ] || { log "build failed rc=$rc"; exit 1; }
cp target/release/memra-server $R/bins/memra-server
sha256sum $R/bins/memra-server > $R/binary.sha256
idle() {
  flock -n /tmp/memra-gpu.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
}
for order in O1 O2; do
  deadline=$((SECONDS + 7200))
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "order $order: card not idle after 7200 s; not run"; exit 3; }
    sleep 30
  done
  log "order $order: collector start"
  rm -rf $R/collector-$order
  env R=$R RIG_LOCK=/tmp/memra-gpu.lock BIN=$R/bins/memra-server WT=/root/wt-b NO_SCOPE=1 BURST=64 \
    MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf MODEL_KEY=q38 \
    python3 tools/tier-battery.py --rig pro-single --timeout 16200 --out $R/collector-$order --external-lock --execute \
    bash research/spill-b-20260919/day31-order.sh @COLLECTOR_LOCK_FD@ $order > $R/collector-$order.log 2>&1
  log "order $order: collector exit=$?"
done
log "chain done"
