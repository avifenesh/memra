#!/usr/bin/env bash
# WP-B DAY39 addendum B target-card half (DAY39.md 1.8): one RTX PRO 6000 Blackwell Workstation Edition, the 27B.
# v3 = B2_SHA (the revision; the DAY38D green file is reused when its source matches), v2 = v3 plus day39b-v2.patch,
# v1 = v2 plus day39-red.patch, built with build-arms.sh; v2-R64, v2-off, v3-R64-r1, v3-off, v3-R64-r2, v1-R64
# (day39-run.sh), then day33-compare.py (unchanged; v2 as red, v3 as green) and day39-read.py. Never a signal to anything
# this lane did not start. Receipts under /root/spill-receipts/b-day39b.
set -uo pipefail
R=/root/spill-receipts/b-day39b; mkdir -p "$R/bins" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
B2_SHA=${B2_SHA:-a803d308080d047eefedf76fc18cf5c09484869e}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
G38=/root/spill-receipts/b-day38d/bins/green
if [ ! -x "$R/bins/v3/memra-server" ] && [ -x "$G38/memra-server" ] \
   && [ "$(cat "$G38/source.commit" 2>/dev/null)" = "$(git rev-parse "$B2_SHA")" ] \
   && ! command grep -q "research/" "$G38/source.patches" 2>/dev/null; then
  mkdir -p "$R/bins/v3"; cp "$G38/memra-server" "$G38/source.commit" "$R/bins/v3/"
  echo "copied from $G38 (the DAY38D green build, same source, no patch)" > "$R/bins/v3/source.patches"
fi
export WT; TARGET=$WT/target WRAP="nice -n 10" bash research/spill-b-20260919/build-arms.sh "$R/bins" "$B2_SHA" v3 v2:day39b-v2.patch \
  v1:day39b-v2.patch+day39-red.patch > "$R/bins/build.out" 2>&1 || { log "builds failed: $(tail -1 "$R/bins/build.out")"; exit 1; }
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS")"
export WT BINS="$R/bins" RIG_LOCK=/tmp/memra-gpu.lock BOOT_CTX='' MODEL MODEL_KEY=q38 NO_SCOPE=1
bash research/spill-b-20260919/day39-run.sh "$R" v2-R64:v2:R64 v2-off:v2:off v3-R64-r1:v3:R64 v3-off:v3:off \
  v3-R64-r2:v3:R64 v1-R64:v1:R64 || { log "boots stopped rc=$?"; exit 1; }
B=$R/boots
python3 research/spill-b-20260919/day33-compare.py --card pro6000 red:R64:$B/v2-R64 red:off:$B/v2-off \
  green:R64:$B/v3-R64-r1 green:off:$B/v3-off green:R64:$B/v3-R64-r2 > "$R/SUMMARY.txt" 2>&1
python3 research/spill-b-20260919/day39-read.py pro6000 $B/v1-R64 $B/v2-R64 $B/v3-R64-r1 $B/v3-R64-r2 $B/v2-off $B/v3-off \
  > "$R/READINGS.txt" 2>&1
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY39B-BOX-DONE"
