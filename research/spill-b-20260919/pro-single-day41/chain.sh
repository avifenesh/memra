#!/usr/bin/env bash
# WP-B DAY41 target-card half (DAY41.md 1.2 to 1.4, addendum A): one RTX PRO 6000 Blackwell Workstation Edition, the 27B
# at the checkpoint's context. Builds the tip's memra-server and concat-prime-probe from S41 and the offprev arm (S41 plus
# day41-nodoor.patch) with build-arms.sh; the probe (6,144 and 30,720, K 32 and 256); the RX and FX boots per route in both
# orders; offprev; the reader; a manifest. Refuses to start without ss or lsof. Never a signal to anything this lane did
# not start. Receipts under /root/spill-receipts/b-day41, mirrored to pro-single-day41/box/.
set -uo pipefail
R=/root/spill-receipts/b-day41; mkdir -p "$R/bins" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
S41=${S41:-99812fe38fd4cb2f021014b34f928794db4fc20c}
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
TARGET=$WT/target WRAP="nice -n 10" bash research/spill-b-20260919/build-arms.sh "$R/bins" "$S41" tip offprev:day41-nodoor.patch \
  > "$R/bins/build.out" 2>&1 || { log "builds failed: $(tail -1 "$R/bins/build.out")"; exit 1; }
if [ ! -x "$R/bins/concat-prime-probe" ]; then
  W=/root/wt-b41-probe
  git worktree add --detach "$W" "$S41" >> "$R/fetch.log" 2>&1 || { log "probe worktree failed"; exit 1; }
  ( cd "$W" && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-engine --bin concat-prime-probe ) \
    > "$R/bins/probe-build.log" 2>&1 || { log "probe build failed"; exit 1; }
  cp /root/wt-b/target/release/concat-prime-probe "$R/bins/"
  git worktree remove --force "$W" >> "$R/fetch.log" 2>&1
fi
sha256sum "$R/bins/concat-prime-probe" >> "$R/bins/SHA256SUMS"
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS" | cut -c1-300)"
RIG_LOCK=/tmp/memra-gpu.lock LENGTHS=6144,30720 bash research/spill-b-20260919/day41-probe.sh "$R/probe" "$R/bins/concat-prime-probe" "$MODEL" \
  > "$R/probe.log" 2>&1
log "probe rc=$?"
export RIG_LOCK=/tmp/memra-gpu.lock BIN="$R/bins/tip/memra-server" PREV_BIN="$R/bins/offprev/memra-server" MODEL MODEL_KEY=q38 \
  BOOT_CTX='' LENGTHS=6144,30720,122880 NO_SCOPE=1
for route in plain spec; do
  bash research/spill-b-20260919/day41-run.sh "$R" "rx-$route-O1-keep:keep:$route:RX" "rx-$route-O1-rewind:rewind:$route:RX" \
    "rx-$route-O2-rewind:rewind:$route:RX" "rx-$route-O2-keep:keep:$route:RX" "fx-$route-keep:keep:$route:FX" \
    "fx-$route-rewind:rewind:$route:FX" || log "boots $route stopped rc=$?"
done
bash research/spill-b-20260919/day41-run.sh "$R" offprev:offprev:plain:RX6 || log "offprev stopped rc=$?"
python3 research/spill-b-20260919/day41-read.py pro6000 "$R" > "$R/read.log" 2>&1
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY41-BOX-DONE"
