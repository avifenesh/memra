#!/usr/bin/env bash
# WP-B day 31 local chain (RTX 5090, Qwen3.5-9B NVFP4 MTP, MEMRA_CTX=65536): one collector hold per order on
# /tmp/memra-5090.lock. Before each hold: a bounded poll (at most 3600 s) until the lock is free, no compute app is
# on the card and host memory has at least 24 GB available; never a signal to anything. The binary is copied once
# so every boot of both orders runs the same bytes.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
R=$WT/research/spill-b-20260919/rtx5090-day31
BIN=$WT/target/day31/memra-server   # under target/ (gitignored): the one binary every boot of both orders runs
mkdir -p "$(dirname "$BIN")"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
[ -f "$BIN" ] || cp target/release/memra-server "$BIN"
sha256sum "$BIN" > "$R/binary.sha256"
git rev-parse HEAD > "$R/source.txt"
log "chain start HEAD=$(git rev-parse HEAD) bin=$(cut -c1-16 "$R/binary.sha256")"
idle() {
  flock -n /tmp/memra-5090.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
for order in O1 O2; do
  deadline=$((SECONDS + 3600))
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "order $order: rig not idle after 3600 s; not run"; exit 3; }
    sleep 30
  done
  log "order $order: collector start"
  rm -rf "$R/collector-$order"
  env R="$R" RIG_LOCK=/tmp/memra-5090.lock BIN="$BIN" WT="$WT" MEMRA_CTX=65536 BURST=32 \
    python3 tools/tier-battery.py --rig rtx5090 --timeout 12600 --out "$R/collector-$order" --external-lock --execute \
    bash research/spill-b-20260919/day31-order.sh @COLLECTOR_LOCK_FD@ "$order" > "$R/collector-$order.log" 2>&1
  log "order $order: collector exit=$?"
done
log "chain done"
