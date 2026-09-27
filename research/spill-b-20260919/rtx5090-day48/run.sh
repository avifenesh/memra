#!/usr/bin/env bash
# DAY48 local half (the 5090, Ornith-1.5-35B-A3B NVFP4 MTP at MEMRA_CTX=65536; DAY48 addendum A): runs from the lane's
# integ71 tree (wt-b-integ71), the lane's crates at S48 built by build-arms.sh at nice 19 under 600% into
# target/day48; then the four boots through day48-run.sh (the lock per boot, YIELD_S=240 after each), burst 32 of
# 6,144 less 256 k, the second wave at the burst's first 200; then day48-read.py. Every exit is captured on its own line.
# Never a signal to anything.
set -uo pipefail
WT=${WT:-$HOME/projects/wt-b-integ71}
D=$WT/research/spill-b-20260919/rtx5090-day48
S48=${S48:-5a6f1898f33bf3c5db68ba149ca6ca73e869f6f5}
MODEL=/data/ai-ml/hf-models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
mkdir -p "$D"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/run.log"; }
W="systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19"
if [ ! -s target/day48/SHA256SUMS ]; then
  echo "$(date -u +%FT%TZ) WP-B DAY48 build (nice 19, CPUQuota=600%)" >> "$WT/research/spill-b-20260919/cpu-concurrency.log"
  export WT; TARGET=$WT/target/arms-build WRAP="$W" bash research/spill-b-20260919/build-arms.sh "$WT/target/day48" "$S48" tip \
    > "$D/build.out" 2>&1
  rc=$?
  [ $rc = 0 ] || { log "build failed rc=$rc"; exit 2; }
fi
cp target/day48/SHA256SUMS "$D/binaries.sha256"
sha256sum "$MODEL" > "$D/model.sha256"
log "start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$D/binaries.sha256") model=$(cut -c1-16 "$D/model.sha256")"
export WT
BIN=$WT/target/day48/tip/memra-server MODEL=$MODEL MODEL_KEY=o15 YIELD_S=240 \
  CLIENT_EXTRA="--burst 32 --length 6144 --max-tokens 64" \
  bash research/spill-b-20260919/day48-run.sh "$D" O1-enforce:enforce O1-enforce-vg:enforce-vg \
  O2-enforce-vg:enforce-vg O2-enforce:enforce
rc=$?
log "boots rc=$rc"
python3 research/spill-b-20260919/day48-read.py rtx5090 "$D" > "$D/read.log" 2>&1
log "done"
