#!/usr/bin/env bash
# DAY37 local chain after A2 (1.8, addendum B): the A1 gate set per arm under one collector hold each, then the serving
# boots (mix both paths both orders, A5 faults, A6 main, A7 bursts, A3 stream pairs both orders). Lock contention is
# retried with a bounded wait; an executed failure is never retried. Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
R=$WT/research/spill-b-20260919/rtx5090-day37
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
log "chain start HEAD=$(git rev-parse HEAD) lane_bin=$(sha256sum target/day37/lane/memra-server | cut -c1-16) main_bin=$(sha256sum target/day37/main/memra-server | cut -c1-16)"
for arm in pooled vmm; do
  out=$R/gates-$arm
  if [ -e "$out/battery.log" ] && command grep -q "gates done arm=$arm" "$out/battery.log"; then log "gates $arm: already done"; continue; fi
  for attempt in $(seq 0 90); do
    rm -rf "$out/collector-$attempt"; mkdir -p "$out"
    python3 tools/tier-battery.py --rig rtx5090 --timeout 7200 --out "$out/collector-$attempt" --external-lock --execute \
      bash research/spill-b-20260919/rtx5090-day37/gates.sh @COLLECTOR_LOCK_FD@ "$arm" "$out" > "$out/collector-$attempt.log" 2>&1
    rc=$?
    if [ $rc = 0 ] || [ -e "$out/battery.log" ]; then log "gates $arm: collector exit=$rc attempt=$attempt"; break; fi
    if ! command grep -q 'Resource temporarily unavailable' "$out/collector-$attempt.log"; then log "gates $arm: collector failed rc=$rc before any cell"; break; fi
    sleep 120
  done
done
B=research/spill-b-20260919/rtx5090-day37/boots.sh
bash $B "$R" \
  mix-spec-O1-pooled:pooled:mixspec mix-spec-O1-vmm:vmm:mixspec mix-plain-O1-pooled:pooled:mixplain mix-plain-O1-vmm:vmm:mixplain \
  mix-spec-O2-vmm:vmm:mixspec mix-spec-O2-pooled:pooled:mixspec mix-plain-O2-vmm:vmm:mixplain mix-plain-O2-pooled:pooled:mixplain \
  fault-mapper:vmm-mapperfault:mixspec off-main:main:mixspec \
  burst-g2-pooled:pooled:g2 burst-g2-vmm:vmm:g2 burst-l64-vmm:vmm:l64 burst-l64-pooled:pooled:l64 burst-boff-pooled:pooled:boff burst-boff-vmm:vmm:boff \
  fault-ensure:vmm-ensurefault:g2 fault-build1:vmm-buildfault1:g2 fault-build64:vmm-buildfault64:g2
args=()
for k in 1 2 3 4 5; do args+=("stream-O1-$k-pooled:pooled:stream" "stream-O1-$k-vmm:vmm:stream"); done
for k in 1 2 3 4 5; do args+=("stream-O2-$k-vmm:vmm:stream" "stream-O2-$k-pooled:pooled:stream"); done
bash $B "$R" "${args[@]}"
python3 research/spill-b-20260919/day37-read.py rtx5090 "$R" > "$R/read.log" 2>&1
log "chain done"
