#!/usr/bin/env bash
# WP-A days 42 to 44 on one RTX PRO 6000 Blackwell on a 9950X-class host, the lead's order: design S2's sitting
# (DAY42), item 15's (DAY43), item 3's 9950X-class reading (DAY44). Every build runs before any cell (no build beside a
# timed cell; each build returns the tree to the tip), then the three drivers in order, each cell under the collector's
# hold. Nothing here is a clause; each sitting's own reader reads it. usage: run-all.sh <tip_sha> <g4_sha>
set -uo pipefail
TIP=$1; G4=$2
mkdir -p /root/spill-receipts
L=/root/spill-receipts/a-run-all.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
cd /root/wt-a || exit 1
D=research/spill-a-20260919
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$TIP")" ] || { log "the clone is not at the tip"; exit 2; }
log "start tip=$TIP g4=$G4 host=$(lscpu | sed -n 's/^Model name: *//p')"
bash $D/pro-single-s2/build.sh "$TIP" "$G4"; log "s2 build rc=$? $(tail -1 /root/spill-receipts/a-s2/build.log 2>&1)"
bash $D/pro-single-i15/build.sh "$TIP" "$G4"; log "i15 build rc=$? $(tail -1 /root/spill-receipts/a-i15/build.log 2>&1)"
bash $D/pro-single-t9950/build.sh "$TIP" "$G4"; log "t9950 build rc=$? $(tail -1 /root/spill-receipts/a-t9950/build.log 2>&1)"
[ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$(git rev-parse "$TIP")" ] || { log "the tree is not the tip after the builds"; exit 2; }
bash $D/pro-single-s2/driver.sh; log "s2 driver rc=$?"
bash $D/pro-single-i15/driver.sh; log "i15 driver rc=$?"
bash $D/pro-single-t9950/driver.sh; log "t9950 driver rc=$?"
log "run-all-done"
