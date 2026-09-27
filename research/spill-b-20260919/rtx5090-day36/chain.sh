#!/usr/bin/env bash
# WP-B day 36 local chain (DAY36.md 1.2, 1.3): RTX 5090, Qwen3.5-9B NVFP4 MTP, MEMRA_CTX=65536. The binary was built
# once from the day-35 fix 809c16444 (rtx5090-day35/build-green/) and copied to target/day36/memra-server; its sha256 is
# binary.sha256 and every boot of both orders runs those bytes. One collector hold per order on /tmp/memra-5090.lock,
# so no other job can run inside an order. Before each hold: a bounded poll (at most 14400 s) until the lock is free,
# no compute app is on the card and host memory has at least 24 GB available; never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
R=$WT/research/spill-b-20260919/rtx5090-day36
BIN=$WT/target/day36/memra-server   # under target/ (gitignored)
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
want=$(cut -d' ' -f1 "$R/binary.sha256"); have=$(sha256sum "$BIN" | cut -d' ' -f1)
[ "$want" = "$have" ] || { log "binary mismatch want=$want have=$have; not run"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
log "chain start HEAD=$(git rev-parse HEAD) bin=${have:0:16} built_from=$(cat "$R/build-source.txt")"
idle() {
  flock -n /tmp/memra-5090.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
for order in O1 O2; do
  deadline=$((SECONDS + 14400)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "order $order: rig not idle after 14400 s; not run"; exit 3; }
    [ $waited = 0 ] && log "order $order: waiting for an idle rig (lock, compute apps, memory)"
    waited=1; sleep 30
  done
  log "order $order: collector start"
  rm -rf "$R/collector-$order"
  env R="$R" RIG_LOCK=/tmp/memra-5090.lock BIN="$BIN" WT="$WT" MEMRA_CTX=65536 BURST=32 \
    python3 tools/tier-battery.py --rig rtx5090 --timeout 12600 --out "$R/collector-$order" --external-lock --execute \
    bash research/spill-b-20260919/day31-order.sh @COLLECTOR_LOCK_FD@ "$order" > "$R/collector-$order.log" 2>&1
  log "order $order: collector exit=$?"
done
log "chain done"
