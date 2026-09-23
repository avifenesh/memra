#!/usr/bin/env bash
# WP-B day 31 D4 local chain (DAY31-D4.md): waits for the day-31 local chain to print "chain done", then one collector
# hold on /tmp/memra-5090.lock for the one D4 boot, after a bounded poll (at most 3600 s) until the lock is free, no
# compute app is on the card and host memory has at least 24 GB available. Same binary as every day-31 local boot.
# Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
R=$WT/research/spill-b-20260919/rtx5090-day31-d4
D3=$WT/research/spill-b-20260919/rtx5090-day31
BIN=$WT/target/day31/memra-server
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
log "chain start HEAD=$(git rev-parse HEAD)"
deadline=$((SECONDS + 21600))
until grep -q "chain done" "$D3/chain.log" 2>/dev/null; do
  [ $SECONDS -ge $deadline ] && { log "day-31 chain not done after 21600 s; D4 not run"; exit 3; }
  sleep 60
done
sha256sum "$BIN" > "$R/binary.sha256"; git rev-parse HEAD > "$R/source.txt"
idle() {
  flock -n /tmp/memra-5090.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
deadline=$((SECONDS + 3600))
until idle; do
  [ $SECONDS -ge $deadline ] && { log "rig not idle after 3600 s; D4 not run"; exit 3; }
  sleep 30
done
log "D4: collector start"
rm -rf "$R/collector-D4"
env R="$R" RIG_LOCK=/tmp/memra-5090.lock BIN="$BIN" WT="$WT" MEMRA_CTX=65536 BURST=32 \
  python3 tools/tier-battery.py --rig rtx5090 --timeout 7200 --out "$R/collector-D4" --external-lock --execute \
  bash research/spill-b-20260919/d4-boot.sh @COLLECTOR_LOCK_FD@ > "$R/collector-D4.log" 2>&1
log "D4: collector exit=$?"
log "chain done"
