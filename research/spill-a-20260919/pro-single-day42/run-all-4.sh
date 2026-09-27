#!/usr/bin/env bash
# WP-A day 48, the box's next run (after run-all-2.sh's): design S4's target sitting (DAY48 section 1: pro-single-s2's
# scripts with S4's tip as the s2 arm; S3's receipts are moved to a-s2-design-s3 first, so each design keeps its own root).
# The build before any cell, then the driver. usage: run-all-4.sh <tip_sha> <g4_sha>
set -uo pipefail
TIP=$1; G4=$2
L=/root/spill-receipts/a-run-all-4.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
cd /root/wt-a || exit 1
D=research/spill-a-20260919
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$TIP")" ] || { log "the clone is not at the tip"; exit 2; }
grep -q 'run-all-2-done' /root/spill-receipts/a-run-all-2.log || { log "the second run has not finished"; exit 2; }
if [ ! -d /root/spill-receipts/a-s2-design-s3 ]; then mv /root/spill-receipts/a-s2 /root/spill-receipts/a-s2-design-s3; fi
log "start tip=$TIP g4=$G4 (S3's receipts at a-s2-design-s3)"
bash $D/pro-single-s2/build.sh "$TIP" "$G4"; log "s4 build rc=$? $(tail -1 /root/spill-receipts/a-s2/build.log 2>&1)"
[ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$(git rev-parse "$TIP")" ] || { log "the tree is not the tip after the build"; exit 2; }
bash $D/pro-single-s2/driver.sh; log "s4 driver rc=$?"
log "run-all-4-done"
