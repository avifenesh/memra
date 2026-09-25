#!/usr/bin/env bash
# WP-B DAY41 addenda B and C, target-card half (DAY41.md 1.8 and 1.9): one RTX PRO 6000 Blackwell Workstation Edition,
# the 27B at the checkpoint's context. Builds the revised tip's memra-server and the offprev arm (the tip plus
# day41b-nodoor.patch) with build-arms.sh; the RX boots per route in both orders; offprev; the reader; a manifest. FX and
# the probe are not rerun (addendum B). Refuses to start without ss or lsof. Never a signal to anything this lane did not
# start. Receipts under /root/spill-receipts/b-day41b, mirrored to pro-single-day41/box-b/.
set -uo pipefail
R=/root/spill-receipts/b-day41b; mkdir -p "$R/bins" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
S41B=${S41B:-23c296b583d0207413d4a2f3347882e729b4b29d}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
command -v ss >/dev/null || command -v lsof >/dev/null || { log "neither ss nor lsof on PATH; not run"; exit 1; }
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
export WT
TARGET=$WT/target WRAP="nice -n 10" bash research/spill-b-20260919/build-arms.sh "$R/bins" "$S41B" tip offprev:day41b-nodoor.patch \
  > "$R/bins/build.out" 2>&1 || { log "builds failed: $(tail -1 "$R/bins/build.out")"; exit 1; }
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS" | cut -c1-300)"
export RIG_LOCK=/tmp/memra-gpu.lock BIN="$R/bins/tip/memra-server" PREV_BIN="$R/bins/offprev/memra-server" MODEL MODEL_KEY=q38 \
  BOOT_CTX='' LENGTHS=6144,30720,122880 NO_SCOPE=1
for route in plain spec; do
  bash research/spill-b-20260919/day41-run.sh "$R" "rx-$route-O1-keep:keep:$route:RX" "rx-$route-O1-rewind:rewind:$route:RX" \
    "rx-$route-O2-rewind:rewind:$route:RX" "rx-$route-O2-keep:keep:$route:RX" || log "boots $route stopped rc=$?"
done
bash research/spill-b-20260919/day41-run.sh "$R" offprev:offprev:plain:RX6 || log "offprev stopped rc=$?"
python3 research/spill-b-20260919/day41-read.py pro6000 "$R" > "$R/read.log" 2>&1
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY41B-BOX-DONE"
