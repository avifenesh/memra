#!/usr/bin/env bash
# WP-B DAY44 target-card half (DAY44.md 1.6, O11 revised: the exact and fast resume): one RTX PRO 6000 Blackwell
# Workstation Edition, the 27B at the checkpoint's context. Builds the tip's memra-server and the offprev arm (the tip
# plus day44-nodoor.patch) with build-arms.sh; per route (plain, spec) and shape (RX zero gap, RXg 1,000 ms gap) the
# keep and exact boots in both orders; offprev; the settle fault boot; the reader; a manifest. Refuses to start without
# ss or lsof. Never a signal to anything this lane did not start. Receipts under /root/spill-receipts/b-day44,
# mirrored to pro-single-day44/box/.
set -uo pipefail
R=/root/spill-receipts/b-day44; mkdir -p "$R/bins" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
S44=${S44:-138790651415cc55a093224e92752ee045861f39}
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
TARGET=$WT/target WRAP="nice -n 10" bash research/spill-b-20260919/build-arms.sh "$R/bins" "$S44" tip offprev:day44-nodoor.patch \
  > "$R/bins/build.out" 2>&1 || { log "builds failed: $(tail -1 "$R/bins/build.out")"; exit 1; }
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS" | cut -c1-300)"
export RIG_LOCK=/tmp/memra-gpu.lock BIN="$R/bins/tip/memra-server" PREV_BIN="$R/bins/offprev/memra-server" MODEL MODEL_KEY=q38 \
  BOOT_CTX='' LENGTHS=6144,30720,122880 NO_SCOPE=1
for route in plain spec; do
  for shape in rx rxg; do
    S=$( [ "$shape" = rx ] && echo RX || echo RXg )
    bash research/spill-b-20260919/day44-run.sh "$R" "rx-$route-$shape-O1-keep:keep:$route:$S" \
      "rx-$route-$shape-O1-exact:exact:$route:$S" "rx-$route-$shape-O2-exact:exact:$route:$S" \
      "rx-$route-$shape-O2-keep:keep:$route:$S" || log "boots $route $shape stopped rc=$?"
  done
done
bash research/spill-b-20260919/day44-run.sh "$R" offprev:offprev:plain:RX6 fault-plain-rxg6:fault:plain:RXg6 \
  || log "offprev/fault stopped rc=$?"
python3 research/spill-b-20260919/day44-read.py pro6000 "$R" > "$R/read.log" 2>&1
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY44-BOX-DONE"
