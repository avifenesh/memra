#!/usr/bin/env bash
# Day 29 target-card chain (one RTX PRO 6000 Blackwell, 27B): fast-forward the lane tip, build, then the two twin
# cells through the collector on the canonical /tmp/memra-gpu.lock (bounded wait when another lane holds it, never a
# signal to anything). Receipts under /root/spill-receipts/b-day29, mirrored to research/spill-b-20260919/pro-single-day29/box/.
set -uo pipefail
R=/root/spill-receipts/b-day29; mkdir -p $R $R/bins
cd /root/wt-b || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> $R/chain.log; }
log "chain start HEAD=$(git rev-parse HEAD)"
git fetch origin lane/spill-b-20260919 >> $R/fetch.log 2>&1 && git merge --ff-only origin/lane/spill-b-20260919 >> $R/fetch.log 2>&1 || { log "fetch/ff failed"; exit 1; }
log "tree HEAD=$(git rev-parse HEAD)"
git rev-parse HEAD > $R/source.txt
( cargo build --release -p memra-server --offline || cargo build --release -p memra-server ) > $R/build.log 2>&1; rc=$?; echo "exit=$rc" >> $R/build.log
[ $rc = 0 ] || { log "build failed rc=$rc"; exit 1; }
cp target/release/memra-server $R/bins/memra-server
sha256sum $R/bins/memra-server > $R/binary.sha256
deadline=$((SECONDS + 3600))
while ! flock -n /tmp/memra-gpu.lock true; do
  [ $SECONDS -ge $deadline ] && { log "lock still held after 3600 s; not run"; exit 3; }
  sleep 30
done
log "gates start"
rm -rf $R/collector
python3 tools/tier-battery.py --rig pro-single --timeout 3600 --out $R/collector --external-lock --execute \
  bash research/spill-b-20260919/pro-single-day29/gates.sh @COLLECTOR_LOCK_FD@ > $R/collector.log 2>&1
log "gates exit=$? ($(cat $R/gates/twin27-off.exit 2>/dev/null) $(cat $R/gates/twin27-on.exit 2>/dev/null))"
log "chain done"
