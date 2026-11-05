#!/usr/bin/env bash
# WP-B DAY46 addendum C target-card rerun (DAY46.md 1.1 to 1.3, addenda A to C, O6; the second wave on the first 200): one RTX PRO 6000 Blackwell Workstation Edition,
# the 27B at the checkpoint's context. The lane's crates at S46 built by build-arms.sh; then the six boots (O1: shadow,
# enforce, enforce-wrel; O2 the reverse) through day46-run.sh under the box lock, burst 64 x 30,720, the second wave at
# the burst's first completion; then day46-read.py. Refuses to start without ss or lsof. Every step's exit is captured
# on its own line. Never a signal to anything this lane did not start. Receipts /root/spill-receipts/b-day46c.
# <box> Max-Q kit port (DEVIATIONS.md D3): R, WT, MODEL, BIN and CARD come from the kit's run.sh; no git fetch, rev-parse
# or cat-file (source.txt is $WT/SOURCE.txt, the chain tree 43dbdb29f; S46 is checked against the binary's
# source.commit); no on-box build (BIN is the laptop-built S46 binary, recorded as bins/tip); the card check reads CARD
# only and accepts the Max-Q name; the reader label is READER_CARD (pro6000maxq). Arms, order, N, prompts and client
# flags unchanged.
set -uo pipefail
R=${R:?receipt root}; mkdir -p "$R/bins" "$R/boots"
# Exported: build-arms.sh and day46-run.sh read WT and RIG_LOCK from the environment (their defaults are the local rig's).
export WT=${WT:?the kit d46 tree} RIG_LOCK=/tmp/memra-gpu.lock
MODEL=${MODEL:?model path}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
S46=${S46:-8926ccfb3e8a27cf8f91aa747e2b088251215c7a}
CARD=${CARD:?CARD}
BIN=${BIN:?the prebuilt S46 memra-server}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
cat "$WT/SOURCE.txt" > "$R/source.txt"
command -v ss >/dev/null || command -v lsof >/dev/null || { log "neither ss nor lsof on PATH; not run"; exit 1; }
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi -i "$CARD" --query-gpu=name,power.limit,memory.total,driver_version --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Max-Q Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Max-Q Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
[ "$(cat "$(dirname "$BIN")/source.commit" 2>/dev/null)" = "$S46" ] || { log "binary $BIN is not built from S46 $S46; not run"; exit 1; }
[ -d "$WT/research/spill-b-20260919" ] || { log "no lane tree at WT=$WT; not run"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
mkdir -p "$R/bins/tip"
echo "$S46" > "$R/bins/tip/source.commit"; : > "$R/bins/tip/source.patches"
echo "prebuilt on the laptop by the kit's build.sh: $BIN (PROVENANCE.txt)" > "$R/bins/tip/prebuilt.txt"
echo "$(sha256sum "$BIN" | cut -d' ' -f1)  ./tip/memra-server" > "$R/bins/SHA256SUMS"
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS" | cut -c1-200)"
export BIN MODEL MODEL_KEY=q38 BOOT_CTX='' NO_SCOPE=1 \
  CLIENT_EXTRA="--burst 64 --length 30720 --max-tokens 64"
bash research/spill-b-20260919/day46-run.sh "$R" O1-shadow:shadow O1-enforce:enforce O1-enforce-wrel:enforce-wrel \
  O2-enforce-wrel:enforce-wrel O2-enforce:enforce O2-shadow:shadow
rc=$?
[ $rc = 0 ] || log "boots stopped rc=$rc"
python3 research/spill-b-20260919/day46-read.py "${READER_CARD:-pro6000maxq}" "$R" > "$R/read.log" 2>&1
( cd "$R" && find . -type f ! -name MANIFEST.sha256 ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY46C-BOX-DONE"
