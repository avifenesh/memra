#!/usr/bin/env bash
# WP-B DAY46 addendum C target-card rerun (DAY46.md 1.1 to 1.3, addenda A to C, O6; the second wave on the first 200): one RTX PRO 6000 Blackwell Workstation Edition,
# the 27B at the checkpoint's context. The lane's crates at S46 built by build-arms.sh; then the six boots (O1: shadow,
# enforce, enforce-wrel; O2 the reverse) through day46-run.sh under the box lock, burst 64 x 30,720, the second wave at
# the burst's first completion; then day46-read.py. Refuses to start without ss or lsof. Every step's exit is captured
# on its own line. Never a signal to anything this lane did not start. Receipts /root/spill-receipts/b-day46c.
set -uo pipefail
R=/root/spill-receipts/b-day46c; mkdir -p "$R/bins" "$R/boots"
# Exported: build-arms.sh and day46-run.sh read WT and RIG_LOCK from the environment (their defaults are the local rig's).
export WT=/root/wt-b RIG_LOCK=/tmp/memra-gpu.lock
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
S46=${S46:-8926ccfb3e8a27cf8f91aa747e2b088251215c7a}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
command -v ss >/dev/null || command -v lsof >/dev/null || { log "neither ss nor lsof on PATH; not run"; exit 1; }
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total,driver_version --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
git cat-file -e "$S46^{commit}" || { log "no commit $S46; not run"; exit 1; }
[ -d "$WT/research/spill-b-20260919" ] || { log "no lane tree at WT=$WT; not run"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
TARGET=$WT/target WRAP="nice -n 10" bash research/spill-b-20260919/build-arms.sh "$R/bins" "$S46" tip > "$R/bins/build.out" 2>&1
rc=$?
[ $rc = 0 ] || { log "build failed rc=$rc: $(tail -1 "$R/bins/build.out")"; exit 1; }
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS" | cut -c1-200)"
export BIN="$R/bins/tip/memra-server" MODEL MODEL_KEY=q38 BOOT_CTX='' NO_SCOPE=1 \
  CLIENT_EXTRA="--burst 64 --length 30720 --max-tokens 64"
bash research/spill-b-20260919/day46-run.sh "$R" O1-shadow:shadow O1-enforce:enforce O1-enforce-wrel:enforce-wrel \
  O2-enforce-wrel:enforce-wrel O2-enforce:enforce O2-shadow:shadow
rc=$?
[ $rc = 0 ] || log "boots stopped rc=$rc"
python3 research/spill-b-20260919/day46-read.py pro6000 "$R" > "$R/read.log" 2>&1
( cd "$R" && find . -type f ! -name MANIFEST.sha256 ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY46C-BOX-DONE"
