#!/usr/bin/env bash
# WP-A day 46, the box's second run (after run-all.sh's): design S3's target sitting (DAY46 section 1: pro-single-s2's
# scripts with S3's tip as the s2 arm; the S2 sitting's receipts are moved to a-s2-design-s2 first, so each design keeps
# its own root) and item 3's 9950X-class reading again (DAY44; attempt 1's build refused the hk patch, the build script
# fixed in 9b3c4e82c). Every build before any cell, then the two drivers. usage: run-all-2.sh <tip_sha> <g4_sha>
set -uo pipefail
TIP=$1; G4=$2
L=/root/spill-receipts/a-run-all-2.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
cd /root/wt-a || exit 1
D=research/spill-a-20260919
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$TIP")" ] || { log "the clone is not at the tip"; exit 2; }
grep -q 'run-all-done' /root/spill-receipts/a-run-all.log || { log "the first run has not finished"; exit 2; }
if [ ! -d /root/spill-receipts/a-s2-design-s2 ]; then mv /root/spill-receipts/a-s2 /root/spill-receipts/a-s2-design-s2; fi
rm -f /root/spill-receipts/a-t9950/BUILD-FAILED
log "start tip=$TIP g4=$G4 (S2's receipts at a-s2-design-s2)"
bash $D/pro-single-s2/build.sh "$TIP" "$G4"; log "s3 build rc=$? $(tail -1 /root/spill-receipts/a-s2/build.log 2>&1)"
bash $D/pro-single-t9950/build.sh "$TIP" "$G4"; log "t9950 build rc=$? $(tail -1 /root/spill-receipts/a-t9950/build.log 2>&1)"
[ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$(git rev-parse "$TIP")" ] || { log "the tree is not the tip after the builds"; exit 2; }
bash $D/pro-single-s2/driver.sh; log "s3 driver rc=$?"
bash $D/pro-single-t9950/driver.sh; log "t9950 driver rc=$?"
log "run-all-2-done"
