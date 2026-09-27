#!/usr/bin/env bash
# WP-B DAY51 target-card half (1.4 and addendum A): one RTX PRO 6000 Blackwell Workstation Edition, the 27B at the
# checkpoint's context. The lane's crates at S51 built by build-arms.sh; then the DAY28 shape walk as the after cell,
# run through the hold runner (rig-hold.sh on the box lock: hold, idle check under the hold, run-day28-cell.sh with
# LOCK=none and the hold's fd closed, release). Refuses to start without ss or lsof. Every step's exit is captured on its
# own line. Never a signal to anything this lane did not start. Receipts /root/spill-receipts/b-day51.
set -uo pipefail
R=${R51:-/root/spill-receipts/b-day51}; mkdir -p "$R/bins" "$R/cells"
export WT=/root/wt-b RIG_LOCK=/tmp/memra-gpu.lock
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
S51=${S51:?the DAY51 commit}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
command -v ss >/dev/null || command -v lsof >/dev/null || { log "neither ss nor lsof on PATH; not run"; exit 1; }
[ -f "$MODEL" ] || { log "model absent: $MODEL; not run"; exit 1; }
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total,driver_version --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
git cat-file -e "$S51^{commit}" || { log "no commit $S51; not run"; exit 1; }
[ -d "$WT/research/spill-b-20260919" ] || { log "no lane tree at WT=$WT; not run"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
TARGET=$WT/target WRAP="nice -n 10" bash research/spill-b-20260919/build-arms.sh "$R/bins" "$S51" tip > "$R/bins/build.out" 2>&1
rc=$?
[ $rc = 0 ] || { log "build failed rc=$rc: $(tail -1 "$R/bins/build.out")"; exit 1; }
BIN=$R/bins/tip/memra-server
log "chain start HEAD=$(cat "$R/source.txt") bin=$(sha256sum "$BIN" | cut -c1-16)"
# shellcheck source=../rig-hold.sh
. research/spill-b-20260919/rig-hold.sh
rig_hold after $((SECONDS + 7200))
r=$?
[ $r = 0 ] || { log "after: rig not held idle within 7200 s; not run"; exit 3; }
RIGDIR=$R/cells MODEL=$MODEL MODEL_KEY=q38 LOCK=none NO_SCOPE=1 PORT=18528 CTX=unset \
  bash research/spill-b-20260919/run-day28-cell.sh after "$BIN" > "$R/cells/after.launch.log" 2>&1 8>&-
rc=$?
rig_release after
log "cell rc=$rc"
( cd "$R" && find . -type f ! -name MANIFEST.sha256 ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY51-BOX-DONE"
