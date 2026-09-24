#!/usr/bin/env bash
# WP-B day 36 target-card chain (DAY36.md 1.2, 1.3), run ON BOX4: the owner's decision cell, the day-34 cell (both
# orders, burst 64) on the day-35 tree. The binary is day 35's green build from 809c16444
# (/root/spill-receipts/b-day35/bins/green); this chain refuses to start unless that build's source and sha256 match its
# own records, the model's sha256 is day 32's and the card is idle. One collector hold per order on /tmp/memra-gpu.lock
# after a bounded idle wait. Never a signal to anything. Receipts under /root/spill-receipts/b-day36, mirrored to
# pro-single-day36/box/.
set -uo pipefail
R=/root/spill-receipts/b-day36; mkdir -p "$R/bins"
SRC=/root/spill-receipts/b-day35/bins/green
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
cd /root/wt-b || { log "no /root/wt-b"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 \
  || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
log "chain start HEAD=$(git rev-parse HEAD)"
[ "$(cat "$SRC/build-source.txt" 2>/dev/null)" = 809c16444ad07aa9edd5d293e7d96d6a91a09fdc ] \
  || { log "day-35 green build source is not 809c16444; not run"; exit 1; }
want=$(cut -d' ' -f1 "$SRC/binary.sha256"); have=$(sha256sum "$SRC/memra-server" | cut -d' ' -f1)
[ "$want" = "$have" ] || { log "day-35 green binary sha256 mismatch want=$want have=$have; not run"; exit 1; }
mkdir -p "$R/bins/tip"
cp "$SRC/memra-server" "$R/bins/tip/memra-server"
cp "$SRC/build-source.txt" "$SRC/build-line.txt" "$SRC/build.log" "$R/bins/tip/"
sha256sum "$R/bins/tip/memra-server" > "$R/bins/tip/binary.sha256"
log "binary ${have:0:16} from 809c16444 (copied from b-day35/bins/green)"
have=$(sha256sum "$MODEL" | cut -d' ' -f1)
[ "$have" = "$WANT_MODEL" ] || { log "model sha256 $have is not day 32's $WANT_MODEL; not run"; exit 1; }
idle() {
  flock -n /tmp/memra-gpu.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
}
export MODEL MODEL_KEY=q38 NO_SCOPE=1 WT=/root/wt-b
for order in O1 O2; do
  deadline=$((SECONDS + 7200))
  until idle; do [ $SECONDS -ge $deadline ] && { log "order $order: card not idle after 7200 s; not run"; exit 3; }; sleep 30; done
  log "order $order: collector start"
  rm -rf "$R/collector-$order"
  env R="$R" RIG_LOCK=/tmp/memra-gpu.lock BIN="$R/bins/tip/memra-server" BURST=64 \
    python3 tools/tier-battery.py --rig pro-single --timeout "${ORDER_TIMEOUT:-43200}" --out "$R/collector-$order" --external-lock --execute \
    bash research/spill-b-20260919/day31-order.sh @COLLECTOR_LOCK_FD@ $order > "$R/collector-$order.log" 2>&1
  log "order $order: collector exit=$?"
done
log "chain done"
