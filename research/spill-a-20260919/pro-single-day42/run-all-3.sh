#!/usr/bin/env bash
# WP-A day 47, the box's run after run-all-4.sh's: design V's target sitting (DAY47 section 1,
# `pro-single-v/`: the pause A/B base against v, the gates with the pause gate, the hit gate, the unit cells). The build
# before any cell, then the driver. usage: run-all-3.sh <tip_sha> <base_sha>
set -uo pipefail
TIP=$1; BASE=$2
L=/root/spill-receipts/a-run-all-3.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
cd /root/wt-a || exit 1
D=research/spill-a-20260919
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$TIP")" ] || { log "the clone is not at the tip"; exit 2; }
grep -q "run-all-4-done" /root/spill-receipts/a-run-all-4.log || { log "the S4 run has not finished"; exit 2; }
log "start tip=$TIP base=$BASE"
bash $D/pro-single-v/build.sh "$TIP" "$BASE"; log "v build rc=$? $(tail -1 /root/spill-receipts/a-v/build.log 2>&1)"
[ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$(git rev-parse "$TIP")" ] || { log "the tree is not the tip after the build"; exit 2; }
bash $D/pro-single-v/driver.sh; log "v driver rc=$?"
log "run-all-3-done"
