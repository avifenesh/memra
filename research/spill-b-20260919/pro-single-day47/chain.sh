#!/usr/bin/env bash
# WP-B DAY47 target-card half (DAY47.md 1.3, addenda A and B, O7): one RTX PRO 6000 Blackwell Workstation Edition, the
# 27B. tools/health-fault-gate.sh's arms g (with g-red and the documented g-batch) and h (with h-red), twice, under the
# box lock; the gate builds its own release binary first. Refuses to start without ss or lsof. Never a signal to
# anything this lane did not start. Receipts under /root/spill-receipts/b-day47, mirrored to pro-single-day47/box/.
set -uo pipefail
R=/root/spill-receipts/b-day47; mkdir -p "$R"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
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
nice -n 10 cargo build --release -p memra-server > "$R/build.log" 2>&1 || { log "build failed"; exit 1; }
sha256sum target/release/memra-server > "$R/binary.sha256"
log "chain start HEAD=$(cat "$R/source.txt") $(cut -c1-16 "$R/binary.sha256")"
for run in b1 b2; do
  HFG_ARMS=g,h HFG_OUT="$R/$run" HFG_READY_WAIT_S=900 flock -w 7200 /tmp/memra-gpu.lock tools/health-fault-gate.sh "$MODEL" \
    > "$R/$run.log" 2>&1
  log "$run rc=$? $(tail -1 "$R/$run.log")"
done
( cd "$R" && find . -type f -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY47-BOX-DONE"
