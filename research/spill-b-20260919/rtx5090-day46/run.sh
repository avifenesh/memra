#!/usr/bin/env bash
# DAY46 local half (the 5090, the 9B at MEMRA_CTX=65536): the lane's crates at S46 built by build-arms.sh at nice 19
# under 600% into target/day46; then the six boots through day46-run.sh (it takes the lock per boot and yields
# YIELD_S=240 after each), burst 32 x 6,144; then day46-read.py. Every exit is captured on its own line. Never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919/rtx5090-day46
S46=${S46:-8926ccfb3e8a27cf8f91aa747e2b088251215c7a}
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/run.log"; }
W="systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19"
if [ ! -s target/day46/SHA256SUMS ]; then
  echo "$(date -u +%FT%TZ) WP-B DAY46 build (nice 19, CPUQuota=600%)" >> "$WT/research/spill-b-20260919/cpu-concurrency.log"
  export WT; TARGET=$WT/target/arms-build WRAP="$W" bash research/spill-b-20260919/build-arms.sh "$WT/target/day46" "$S46" tip \
    > "$D/build.out" 2>&1
  rc=$?
  [ $rc = 0 ] || { log "build failed rc=$rc"; exit 2; }
fi
cp target/day46/SHA256SUMS "$D/binaries.sha256"
log "start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$D/binaries.sha256")"
BIN=$WT/target/day46/tip/memra-server YIELD_S=240 CLIENT_EXTRA="--burst 32 --length 6144 --max-tokens 64" \
  bash research/spill-b-20260919/day46-run.sh "$D" O1-shadow:shadow O1-enforce:enforce O1-enforce-wrel:enforce-wrel \
  O2-enforce-wrel:enforce-wrel O2-enforce:enforce O2-shadow:shadow
rc=$?
log "boots rc=$rc"
python3 research/spill-b-20260919/day46-read.py rtx5090 "$D" > "$D/read.log" 2>&1
log "done"
