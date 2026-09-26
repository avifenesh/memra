#!/usr/bin/env bash
# WP-B DAY49 addendum C target-card half (the i-vmm cell): one RTX PRO 6000 Blackwell Workstation Edition, the 27B.
# tools/health-fault-gate.sh's arm i (i-ctrl, i, i-red) under MEMRA_KV_ALLOCATOR=vmm, run from two detached worktrees
# built first: green at the fix (95d35c383), red at 02dbdfa40; each under the box lock; then day49c-read.py. Refuses to
# start without ss or lsof. Never a signal to anything this lane did not start. Receipts under
# /root/spill-receipts/b-day49c, mirrored to pro-single-day49c/box/.
set -uo pipefail
R=/root/spill-receipts/b-day49c; mkdir -p "$R"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
SIDES=(green:95d35c3838853056d02ec356a06add1ed180d320 red:02dbdfa408a5d0215248c0ec2849cd599ca64083)
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
command -v ss >/dev/null || command -v lsof >/dev/null || { log "neither ss nor lsof on PATH; not run"; exit 1; }
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
for s in "${SIDES[@]}"; do git cat-file -e "${s#*:}^{commit}" || { log "no commit ${s#*:}; not run"; exit 1; }; done
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
: > "$R/binaries.sha256"
for s in "${SIDES[@]}"; do
  name=${s%%:*}; sha=${s#*:}; T=$WT/target/wt-day49c-$name
  [ -d "$T" ] || git worktree add --detach "$T" "$sha" > "$R/$name-worktree.log" 2>&1 || { log "$name: worktree failed"; exit 1; }
  ( cd "$T" && nice -n 10 cargo build --release -p memra-server ) > "$R/$name-build.log" 2>&1 || { log "$name: build failed"; exit 1; }
  echo "$sha $(sha256sum "$T/target/release/memra-server" | cut -d' ' -f1) $name" >> "$R/binaries.sha256"
done
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/binaries.sha256" | cut -c1-220)"
for s in "${SIDES[@]}"; do
  name=${s%%:*}; T=$WT/target/wt-day49c-$name
  MEMRA_KV_ALLOCATOR=vmm HFG_ARMS=i HFG_OUT="$R/$name" HFG_READY_WAIT_S=900 flock -w 7200 /tmp/memra-gpu.lock \
    "$T/tools/health-fault-gate.sh" "$MODEL" > "$R/$name.log" 2>&1
  log "$name rc=$? $(tail -1 "$R/$name.log")"
done
python3 research/spill-b-20260919/day49c-read.py pro6000 "$R/green" "$R/red" > "$R/read.log" 2>&1
for s in "${SIDES[@]}"; do git worktree remove --force "$WT/target/wt-day49c-${s%%:*}" >> "$R/chain.log" 2>&1; done
( cd "$R" && find . -type f -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY49C-BOX-DONE"
