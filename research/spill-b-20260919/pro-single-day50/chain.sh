#!/usr/bin/env bash
# WP-B DAY50 stage 0 target-card half (DAY50.md 1.2): one RTX PRO 6000 Blackwell Workstation Edition, the 27B.
# concat-prime-probe built from S50, then day50-stage0.sh at L = 6,144, 30,720 and 122,880 (the wall run, then the nsys
# run, per L, under the box lock). Refuses to start without nsys. Every step's exit is captured on its own line. Never
# a signal to anything this lane did not start. Receipts /root/spill-receipts/b-day50.
set -uo pipefail
R=/root/spill-receipts/b-day50; mkdir -p "$R"
export WT=/root/wt-b RIG_LOCK=/tmp/memra-gpu.lock
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
S50=${S50:?the stage-0 probe commit}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
command -v nsys >/dev/null || { log "no nsys on PATH; not run"; exit 1; }
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total,driver_version --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
git cat-file -e "$S50^{commit}" || { log "no commit $S50; not run"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done (nsys $(nsys --version 2>&1 | tail -1))"; exit 0; }
T=$WT/target/wt-day50
[ -d "$T" ] || git worktree add --detach "$T" "$S50" > "$R/worktree.log" 2>&1 || { log "worktree failed"; exit 1; }
( cd "$T" && nice -n 10 cargo build --release -p memra-engine --bin concat-prime-probe ) > "$R/build.log" 2>&1
rc=$?
[ $rc = 0 ] || { log "build failed rc=$rc"; exit 1; }
sha256sum "$T/target/release/concat-prime-probe" | sed "s|$T/||" > "$R/probe.sha256"
log "chain start HEAD=$(cat "$R/source.txt") S50=$S50 probe=$(cut -c1-16 "$R/probe.sha256")"
LENGTHS=6144,30720,122880 YIELD_S=5 bash research/spill-b-20260919/day50-stage0.sh "$R/stage0" \
  "$T/target/release/concat-prime-probe" "$MODEL" pro6000
rc=$?
log "stage0 rc=$rc"
git worktree remove --force "$T" >> "$R/chain.log" 2>&1
( cd "$R" && find . -type f ! -name '*.nsys-rep' ! -name '*.sqlite' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY50-S0-BOX-DONE"
