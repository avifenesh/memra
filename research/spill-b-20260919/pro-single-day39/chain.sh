#!/usr/bin/env bash
# WP-B day 39 target-card half (DAY39.md 1.3, 1.4): one RTX PRO 6000 Blackwell Workstation Edition, the 27B at the
# checkpoint's context. Builds red and green with day39-build.sh (green = the O5 commit, red = green plus
# day39-red.patch), then red-R64, red-off, green-R64-r1, green-off, green-R64-r2, each under /tmp/memra-gpu.lock after
# a bounded idle wait, then day33-compare.py (unchanged), day39-read.py and a receipts manifest. Never a signal to
# anything this lane did not start. Receipts under /root/spill-receipts/b-day39, mirrored to pro-single-day39/box/.
set -uo pipefail
R=/root/spill-receipts/b-day39; mkdir -p "$R/bins" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && log "dry run: the checks only"
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell: $(tail -1 "$R/card.csv")"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
export WT; TARGET=$WT/target WRAP="nice -n 10" bash research/spill-b-20260919/day39-build.sh "$R/bins" > "$R/bins/build.out" 2>&1 \
  || { log "builds failed: $(tail -1 "$R/bins/build.out")"; exit 1; }
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS")"
export WT BINS="$R/bins" RIG_LOCK=/tmp/memra-gpu.lock BOOT_CTX='' MODEL MODEL_KEY=q38 NO_SCOPE=1
bash research/spill-b-20260919/day39-run.sh "$R" red-R64:red:R64 red-off:red:off green-R64-r1:green:R64 green-off:green:off \
  green-R64-r2:green:R64 || { log "boots stopped rc=$?"; exit 1; }
B=$R/boots
python3 research/spill-b-20260919/day33-compare.py --card pro6000 red:R64:$B/red-R64 red:off:$B/red-off \
  green:R64:$B/green-R64-r1 green:off:$B/green-off green:R64:$B/green-R64-r2 > "$R/SUMMARY.txt" 2>&1
python3 research/spill-b-20260919/day39-read.py pro6000 $B/red-R64 $B/green-R64-r1 $B/green-R64-r2 $B/red-off $B/green-off \
  > "$R/READINGS.txt" 2>&1
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY39-BOX-DONE"
