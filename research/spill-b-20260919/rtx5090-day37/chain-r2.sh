#!/usr/bin/env bash
# DAY37 local chain on the revised binary (addendum C): A2 again (grow-32768-r2), the gate set per arm again
# (gates-r2-*, twin on the 27B), then the serving boots of addendum B. Same rules as chain.sh.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
R=$WT/research/spill-b-20260919/rtx5090-day37
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain-r2.log"; }
log "chain-r2 start HEAD=$(git rev-parse HEAD) server=$(sha256sum target/day37/r2/memra-server | cut -c1-16) gate=$(sha256sum target/day37/r2/kv-tier-gate | cut -c1-16) main=$(sha256sum target/day37/main/memra-server | cut -c1-16)"
if [ ! -e "$R/grow-32768-r2/collector.exit" ]; then
  GATE_BIN=$WT/target/day37/r2/kv-tier-gate bash research/spill-b-20260919/rtx5090-day37/run-grow.sh rtx5090 grow-32768-r2 >> "$R/chain-r2.log" 2>&1
  log "A2 r2 rc=$? $(head -1 "$R/grow-32768-r2/receipt/GROW.txt" 2>/dev/null)"
fi
export BIN=$WT/target/day37/r2/memra-server
for arm in pooled vmm; do
  out=$R/gates-r2-$arm
  if [ -e "$out/battery.log" ] && command grep -q "gates done arm=$arm" "$out/battery.log"; then log "gates r2 $arm: already done"; continue; fi
  for attempt in $(seq 0 90); do
    rm -rf "$out/collector-$attempt"; mkdir -p "$out"
    python3 tools/tier-battery.py --rig rtx5090 --timeout 7200 --out "$out/collector-$attempt" --external-lock --execute \
      bash research/spill-b-20260919/rtx5090-day37/gates.sh @COLLECTOR_LOCK_FD@ "$arm" "$out" > "$out/collector-$attempt.log" 2>&1
    rc=$?
    if [ $rc = 0 ] || [ -e "$out/battery.log" ]; then log "gates r2 $arm: collector exit=$rc attempt=$attempt"; break; fi
    if ! command grep -q 'Resource temporarily unavailable' "$out/collector-$attempt.log"; then log "gates r2 $arm: collector failed rc=$rc before any cell"; break; fi
    sleep 120
  done
done
export LANE_BIN=$WT/target/day37/r2/memra-server
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
python3 research/spill-b-20260919/day37-read.py rtx5090 "$R" > "$R/read-r2.log" 2>&1
log "chain-r2 done"
