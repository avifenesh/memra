#!/usr/bin/env bash
# WP-B day 40 local chain (DAY40.md 1.2): RTX 5090, Qwen3.5-9B NVFP4 MTP, MEMRA_CTX=65536, burst 32; day 36's local
# chain on S40's binary (target/day40/tip/memra-server, built with build-arms.sh; its sha256 is binaries.sha256). One
# collector hold per order on /tmp/memra-5090.lock after a bounded idle poll; never a signal to anything. Then
# day34-compare.py (unchanged) and day31-faults.py.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R=$D/rtx5090-day40
BIN=$WT/target/day40/tip/memra-server
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
sha256sum "$BIN" > "$R/binary.sha256"; cp "$WT/target/day40/tip/source.commit" "$R/build-source.txt"
log "chain start HEAD=$(git rev-parse HEAD) bin=$(cut -c1-16 "$R/binary.sha256") built_from=$(cat "$R/build-source.txt")"
idle() {
  flock -n /tmp/memra-5090.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
for order in O1 O2; do
  deadline=$((SECONDS + 14400)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "order $order: rig not idle after 14400 s; not run"; exit 3; }
    [ $waited = 0 ] && log "order $order: waiting for an idle rig"
    waited=1; sleep 30
  done
  log "order $order: collector start"
  rm -rf "$R/collector-$order"
  env R="$R" RIG_LOCK=/tmp/memra-5090.lock BIN="$BIN" WT="$WT" MEMRA_CTX=65536 BURST=32 \
    python3 tools/tier-battery.py --rig rtx5090 --timeout 12600 --out "$R/collector-$order" --external-lock --execute \
    bash research/spill-b-20260919/day31-order.sh @COLLECTOR_LOCK_FD@ "$order" > "$R/collector-$order.log" 2>&1
  log "order $order: collector exit=$?"
  # The lane yields the card between orders when YIELD_S is set (lead, 2026-09-26): an order is one hold.
  sleep "${YIELD_S:-0}"
done
B=research/spill-b-20260919/rtx5090-day40/boots
( cd "$WT" && python3 "$D/day34-compare.py" --card rtx5090 --served-ctx 65536 --model-ctx 262144 --registry-value 32768 \
    --survey context O1:$B/O1-off O1:$B/O1-on2048 O1:$B/O1-on8192 O1:$B/O1-on32768 O2:$B/O2-off O2:$B/O2-on2048 \
    O2:$B/O2-on8192 O2:$B/O2-on32768 > "$R/SUMMARY.txt" 2>&1 )
# the lister takes each boot's directory (the target card's first run passed the boots root; DAY40 2.1)
python3 "$D/day31-faults.py" "$R"/boots/O*/ > "$R/FAULTS.txt" 2>&1
python3 "$D/day40-read.py" rtx5090 "$R/boots" > "$R/READINGS.txt" 2>&1
log "chain done"
