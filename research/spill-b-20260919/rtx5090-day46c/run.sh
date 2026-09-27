#!/usr/bin/env bash
# DAY46 addendum C local rerun (the 5090, the 9B at MEMRA_CTX=65536): the second wave on the burst's first 200. It runs
# from the lane's integ71 tree (the fixed day46-client.py lives there; the live checkout still carries the old one)
# with the binary DAY46's local half built at S46 (8926ccfb3) into wt-spill-b/target/day46 (build-arms.sh, nice 19,
# 600%), rebuilt here if absent. Six boots through day46-run.sh (the lock per boot, YIELD_S=240 after each), then
# day46-read.py. Every exit is captured on its own line. Never a signal to anything.
set -uo pipefail
WT=${WT:-$HOME/projects/wt-b-integ71}
D=$WT/research/spill-b-20260919/rtx5090-day46c
S46=${S46:-8926ccfb3e8a27cf8f91aa747e2b088251215c7a}
BINS=${BINS:-$HOME/projects/wt-spill-b/target/day46}
mkdir -p "$D"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/run.log"; }
W="systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19"
if [ ! -s "$BINS/SHA256SUMS" ] || [ "$(cat "$BINS/tip/source.commit" 2>/dev/null)" != "$S46" ]; then
  BINS=$WT/target/day46
  echo "$(date -u +%FT%TZ) WP-B DAY46C build (nice 19, CPUQuota=600%)" >> "$WT/research/spill-b-20260919/cpu-concurrency.log"
  export WT; TARGET=$WT/target/arms-build WRAP="$W" bash research/spill-b-20260919/build-arms.sh "$BINS" "$S46" tip \
    > "$D/build.out" 2>&1
  rc=$?
  [ $rc = 0 ] || { log "build failed rc=$rc"; exit 2; }
fi
cp "$BINS/SHA256SUMS" "$D/binaries.sha256"; cp "$BINS/tip/source.commit" "$D/binary.source"
log "start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$D/binaries.sha256") source=$(cat "$D/binary.source")"
export WT
BIN=$BINS/tip/memra-server YIELD_S=240 CLIENT_EXTRA="--burst 32 --length 6144 --max-tokens 64" \
  bash research/spill-b-20260919/day46-run.sh "$D" O1-shadow:shadow O1-enforce:enforce O1-enforce-wrel:enforce-wrel \
  O2-enforce-wrel:enforce-wrel O2-enforce:enforce O2-shadow:shadow
rc=$?
log "boots rc=$rc"
python3 research/spill-b-20260919/day46-read.py rtx5090 "$D" > "$D/read.log" 2>&1
log "done"
