#!/usr/bin/env bash
# WP-B day 38 target-card half (DAY38.md 1.3, addenda A and B): one RTX PRO 6000 Blackwell WS, the 27B at the
# checkpoint's context, after the day-37 sitting on the same box. Builds green and red from the 5090's day-38 source,
# pinned (GREEN_SHA, the tree target/day38 was built from; red = green plus day38-red.patch), each in a detached
# worktree, then the same boots as the local chain with the target card's lengths, the reader, and a receipts
# manifest. Never a signal to anything this lane did not start.
set -uo pipefail
R=/root/spill-receipts/b-day38; mkdir -p "$R/bins/green" "$R/bins/red" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
GREEN_SHA=${GREEN_SHA:-99fef8898}
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell: $(tail -1 "$R/card.csv")"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
if [ ! -x "$R/bins/green/memra-server" ]; then
  W=/root/wt-b38-green
  git worktree add --detach "$W" "$GREEN_SHA" >> "$R/fetch.log" 2>&1 || { log "green worktree failed"; exit 1; }
  ( cd "$W" && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server --bin memra-server ) \
    > "$R/bins/green/build.log" 2>&1 || { log "green build failed"; exit 1; }
  cp /root/wt-b/target/release/memra-server "$R/bins/green/memra-server"; git -C "$W" rev-parse HEAD > "$R/bins/green/source.commit"
  git worktree remove --force "$W" >> "$R/fetch.log" 2>&1
fi
if [ ! -x "$R/bins/red/memra-server" ]; then
  W=/root/wt-b38-red
  git worktree add --detach "$W" "$GREEN_SHA" >> "$R/fetch.log" 2>&1 || { log "red worktree failed"; exit 1; }
  ( cd "$W" && git apply research/spill-b-20260919/day38-red.patch && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 \
      cargo build --release -p memra-server --bin memra-server ) > "$R/bins/red/build.log" 2>&1 || { log "red build failed"; exit 1; }
  cp /root/wt-b/target/release/memra-server "$R/bins/red/memra-server"
  { git -C "$W" rev-parse HEAD; sha256sum "$W/research/spill-b-20260919/day38-red.patch"; } > "$R/bins/red/source.commit"
  git worktree remove --force "$W" >> "$R/fetch.log" 2>&1
fi
sha256sum "$R"/bins/*/memra-server > "$R/bins/SHA256SUMS"
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS" | cut -c1-200)"
export WT RIG_LOCK=/tmp/memra-gpu.lock BIN="$R/bins/green/memra-server" RED_BIN="$R/bins/red/memra-server" MODEL MODEL_KEY=q38 \
  BOOT_CTX='' LENGTHS=6144,30720,122880 NO_SCOPE=1
bash research/spill-b-20260919/day38-run.sh "$R" \
  main-O1-off:off main-O1-on:on main-O2-on:on main-O2-off:off \
  fault-batch:on-fault-batch fault-nobatch:on-fault-nobatch fault-nobatch-red:on-fault-nobatch-red \
  vmm-off:vmm-off vmm-on:vmm-on
python3 research/spill-b-20260919/day38-read.py pro6000 "$R" > "$R/read.log" 2>&1
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY38-BOX-DONE"
