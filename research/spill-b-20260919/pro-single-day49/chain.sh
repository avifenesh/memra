#!/usr/bin/env bash
# WP-B DAY49 target-card half (DAY49.md 1.3 and addendum A, O14): one RTX PRO 6000 Blackwell Workstation Edition, the
# 27B. tools/health-fault-gate.sh's arm i (i-ctrl, i, i-red) twice under the box lock (the gate builds its own release
# binary first), then the serving shape: day45-client.py's burst of 8 x 6,144 tokens with no warm requests and
# MEMRA_STEP_OOM_FAULT=1, arms off and on (MEMRA_BATCH_OOM_RECOVER=1) in both orders on the tip built by build-arms.sh.
# Refuses to start without ss or lsof. Never a signal to anything this lane did not start. Receipts under
# /root/spill-receipts/b-day49, mirrored to pro-single-day49/box/.
set -uo pipefail
R=/root/spill-receipts/b-day49; mkdir -p "$R/bins" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
S49=${S49:-6102fb63a}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
S49=$(git rev-parse "$S49")
command -v ss >/dev/null || command -v lsof >/dev/null || { log "neither ss nor lsof on PATH; not run"; exit 1; }
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
export WT
TARGET=$WT/target WRAP="nice -n 10" bash research/spill-b-20260919/build-arms.sh "$R/bins" "$S49" tip \
  > "$R/bins/build.out" 2>&1 || { log "builds failed: $(tail -1 "$R/bins/build.out")"; exit 1; }
nice -n 10 cargo build --release -p memra-server > "$R/gate-build.log" 2>&1 || { log "gate build failed"; exit 1; }
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS" | cut -c1-200)"
for run in g1 g2; do
  HFG_ARMS=i HFG_OUT="$R/$run" HFG_READY_WAIT_S=900 flock -w 7200 /tmp/memra-gpu.lock tools/health-fault-gate.sh "$MODEL" \
    > "$R/$run.log" 2>&1
  log "gate $run rc=$? $(tail -1 "$R/$run.log")"
done
export RIG_LOCK=/tmp/memra-gpu.lock BIN="$R/bins/tip/memra-server" MODEL MODEL_KEY=q38 BOOT_CTX='' NO_SCOPE=1 \
  CLIENT_EXTRA="--warm-n 0 --burst 8 --length 6144 --max-tokens 64"
bash research/spill-b-20260919/day49-run.sh "$R" O1-off:off O1-on:on O2-on:on O2-off:off || log "boots stopped rc=$?"
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY49-BOX-DONE"
