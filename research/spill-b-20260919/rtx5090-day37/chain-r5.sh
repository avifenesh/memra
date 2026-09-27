#!/usr/bin/env bash
# DAY37 addendum H local chain on the r5 binaries (build-r5.sh), receipts under rtx5090-day37/r5/: A2 (grow-32768-r5),
# the gate set per arm (gates-r5-*, one collector hold each, its serve-smoke worktree built before the hold), r4's boot
# list one boot per call, the stream pairs one pair per call, then the reader. The lock stays free for YIELD_S after
# A2, after each gate arm, after each boot and after each stream pair (the lead's card sharing). Every GPU step waits
# for an idle rig and holds /tmp/memra-5090.lock alone. Patience (env): ATTEMPTS (run-grow.sh, 120 s apart), GATE_ATTEMPTS
# (each gate arm's collector, 120 s apart), BOOT_TRIES (a boot the idle wait did not run). Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R=$D/rtx5090-day37/r5
YIELD_S=${YIELD_S:-240}
mkdir -p "$R"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
[ -x target/day37/r5/memra-server ] && [ -x target/day37/r5/kv-tier-gate ] && [ -x target/day37/r5main/main/memra-server ] \
  || { log "chain-r5: r5 binaries missing; not run"; exit 2; }
SRC=$(cat target/day37/r5/source.commit)
log "chain-r5 start HEAD=$(git rev-parse HEAD) server=$(sha256sum target/day37/r5/memra-server | cut -c1-16) gate=$(sha256sum target/day37/r5/kv-tier-gate | cut -c1-16) main=$(sha256sum target/day37/r5main/main/memra-server | cut -c1-16) source=$SRC YIELD_S=$YIELD_S"
if [ ! -e "$R/grow-32768-r5/collector.exit" ]; then
  ROOT=$R GATE_BIN=$WT/target/day37/r5/kv-tier-gate bash "$D/rtx5090-day37/run-grow.sh" rtx5090 grow-32768-r5 >> "$R/chain.log" 2>&1
  log "A2 r5 rc=$? $(head -1 "$R/grow-32768-r5/receipt/GROW.txt" 2>/dev/null)"
  sleep "$YIELD_S"
fi
export BIN=$WT/target/day37/r5/memra-server
SMOKE_DIR=$WT/target/smoke-wt-${SRC:0:12}
for arm in pooled vmm; do
  out=$R/gates-r5-$arm
  if [ -e "$out/battery.log" ] && command grep -q "gates done arm=$arm" "$out/battery.log"; then log "gates r5 $arm: already done"; continue; fi
  # The serve-smoke tree gates.sh uses, built before the hold (gates.sh removes it after each arm).
  [ -d "$SMOKE_DIR" ] || git worktree add --detach "$SMOKE_DIR" "$SRC" > "$R/smoke-wt-$arm.log" 2>&1
  echo "$(date -u +%FT%TZ) WP-B day37 r5 serve-smoke prebuild $arm (nice 19, CPUQuota=600%)" >> "$D/cpu-concurrency.log"
  ( cd "$SMOKE_DIR" && systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19 \
      cargo build --release -p memra-server ) >> "$R/smoke-wt-$arm.log" 2>&1
  log "gates r5 $arm: serve-smoke tree prebuilt rc=$?"
  for attempt in $(seq 0 "${GATE_ATTEMPTS:-90}"); do
    rm -rf "$out/collector-$attempt"; mkdir -p "$out"
    python3 tools/tier-battery.py --rig rtx5090 --timeout 7200 --out "$out/collector-$attempt" --external-lock --execute \
      bash "$D/rtx5090-day37/gates.sh" @COLLECTOR_LOCK_FD@ "$arm" "$out" > "$out/collector-$attempt.log" 2>&1
    rc=$?
    if [ $rc = 0 ] || [ -e "$out/battery.log" ]; then log "gates r5 $arm: collector exit=$rc attempt=$attempt"; break; fi
    if ! command grep -q 'Resource temporarily unavailable' "$out/collector-$attempt.log"; then log "gates r5 $arm: collector failed rc=$rc before any cell"; break; fi
    sleep 120
  done
  sleep "$YIELD_S"
done
export LANE_BIN=$WT/target/day37/r5/memra-server MAIN_BIN=$WT/target/day37/r5main/main/memra-server
B=$D/rtx5090-day37/boots.sh
done_boot() { command grep -q "boot $1 rc=" "$R/run.log" 2>/dev/null; }
# boots.sh exits 3 when a boot did not run (the idle wait ran out); that boot is not a result, so it is asked again, up
# to six times. An executed boot is never rerun.
run_boots() { # <spec>...
  local try spec left
  for try in $(seq 1 "${BOOT_TRIES:-6}"); do
    left=(); for spec in "$@"; do done_boot "${spec%%:*}" || left+=("$spec"); done
    [ ${#left[@]} = 0 ] && return 0
    [ $try = 1 ] || log "boots ${left[*]}: asked again (try $try) after an idle-wait timeout"
    bash "$B" "$R" "${left[@]}"
    [ $? = 3 ] || return 0
  done
  log "boots ${left[*]}: not run after ${BOOT_TRIES:-6} idle waits"
}
for spec in mix-spec-O1-pooled:pooled:mixspec mix-spec-O1-vmm:vmm:mixspec mix-plain-O1-pooled:pooled:mixplain \
  mix-plain-O1-vmm:vmm:mixplain mix-spec-O2-vmm:vmm:mixspec mix-spec-O2-pooled:pooled:mixspec mix-plain-O2-vmm:vmm:mixplain \
  mix-plain-O2-pooled:pooled:mixplain fault-mapper:vmm-mapperfault:mixspec off-main:main:mixspec \
  burst-g2-pooled:pooled:g2 burst-g2-vmm:vmm:g2 burst-l64-vmm:vmm:l64 burst-l64-pooled:pooled:l64 \
  burst-boff-pooled:pooled:boff burst-boff-vmm:vmm:boff fault-ensure:vmm-ensurefault:g2 \
  fault-build1:vmm-buildfault1:g2 fault-build64:vmm-buildfault64:g2; do
  done_boot "${spec%%:*}" && continue
  run_boots "$spec"
  sleep "$YIELD_S"
done
for k in 1 2 3 4 5; do
  done_boot "stream-O1-$k-vmm" || { run_boots "stream-O1-$k-pooled:pooled:stream" "stream-O1-$k-vmm:vmm:stream"; sleep "$YIELD_S"; }
done
for k in 1 2 3 4 5; do
  done_boot "stream-O2-$k-pooled" || { run_boots "stream-O2-$k-vmm:vmm:stream" "stream-O2-$k-pooled:pooled:stream"; sleep "$YIELD_S"; }
done
python3 "$D/day37-read.py" rtx5090 "$R" gates-r5 > "$R/read.log" 2>&1
log "chain-r5 done"
