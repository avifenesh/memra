#!/usr/bin/env bash
# WP-B day 32 target-card chain (DAY32.md 1.2, 1.3): one RTX PRO 6000 Blackwell, Qwen3.8-27B NVFP4-Q5K MTP, MEMRA_CTX
# unset. The binary was built once from main d544c6b82 in a detached worktree (b-day32/build-source.txt,
# build-line.txt, build.log) and copied to b-day32/bins/; every boot of both orders runs those bytes. This script
# fast-forwards the lane tip for the drivers only, then takes one collector hold per order on /tmp/memra-gpu.lock.
# Before each hold: a bounded poll (at most 7200 s) until the lock is free and no compute app is on the card; never a
# signal to anything. Receipts under /root/spill-receipts/b-day32, mirrored to pro-single-day32/box/.
set -uo pipefail
# RUN names a whole-order rerun (DAY32.md 1.8); ORDER_TIMEOUT is the collector hold per order in seconds.
B=/root/spill-receipts/b-day32
R=$B${RUN:+-$RUN}; mkdir -p "$R"
BIN=$B/bins/memra-server
cd /root/wt-b || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
log "chain start HEAD=$(git rev-parse HEAD) RUN=${RUN:-} ORDER_TIMEOUT=${ORDER_TIMEOUT:-43200}"
grep -qx 'exit=0' "$B/build.log" || { log "build not finished or failed; not run"; exit 1; }
want=$(cut -d' ' -f1 "$B/binary.sha256"); have=$(sha256sum "$BIN" | cut -d' ' -f1)
[ "$want" = "$have" ] || { log "binary mismatch want=$want have=$have; not run"; exit 1; }
git fetch origin lane/spill-b-20260919 >> "$R/fetch.log" 2>&1 && git merge --ff-only FETCH_HEAD >> "$R/fetch.log" 2>&1 \
  || { log "fetch/ff failed"; exit 1; }
log "tree HEAD=$(git rev-parse HEAD) bin=${have:0:16} built_from=$(cat "$B/build-source.txt")"
git rev-parse HEAD > "$R/source.txt"
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
  rm -rf "$R/collector-$order"
  env R="$R" RIG_LOCK=/tmp/memra-gpu.lock BIN="$BIN" WT=/root/wt-b NO_SCOPE=1 BURST=64 \
    MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf MODEL_KEY=q38 \
    python3 tools/tier-battery.py --rig pro-single --timeout "${ORDER_TIMEOUT:-43200}" --out "$R/collector-$order" --external-lock --execute \
    bash research/spill-b-20260919/day31-order.sh @COLLECTOR_LOCK_FD@ $order > "$R/collector-$order.log" 2>&1
  log "order $order: collector exit=$?"
done
log "chain done"
