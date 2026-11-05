#!/usr/bin/env bash
# kv-vmm Max-Q cell, the untimed pass/fail half (DAY37 1.8 step 6 order with the timed steps moved to chain-timed.sh):
#   A2 (the grow series), A1 gate set per arm (one collector hold each), then the 19 untimed serving boots of the two
#   Workstation sittings as one sitting: the mix both paths both orders (A1-MIX, A4), off-lane, fault-mapper, off-main
#   (A5-MAPPER, A6), the bursts (A7), fault-ensure, fault-build1, fault-build64 (A5). The boot list and order are
#   chain.sh:100-105 with off-lane before fault-mapper as rerun-f.sh:64 ran them; every boot through boots-f.sh.
# Run only by run.sh (it holds the window, the canonical lock on FD 7 and the run marker). Steps already done in OUT
# are skipped; a step a deadline cut is moved to OUT/cut/ and asked again (DEVIATIONS.md).
set -uo pipefail
: "${OUT:?}" "${K:?}" "${CARD:?}" "${KIT_HARNESS:?}" "${KIT_LOCK_FD:?}" "${MODEL:?}"
R=$OUT; WT=$KIT_HARNESS; B=$K/bins
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
idle() {
  python3 tools/tier-lock-proof.py --fd "$KIT_LOCK_FD" --lock /tmp/memra-gpu.lock > /dev/null 2>&1 || return 1
  [ -z "$(nvidia-smi -i "$CARD" --query-compute-apps=pid --format=csv,noheader)" ] || return 1
}
wait_idle() {
  local deadline=$((SECONDS + KIT_IDLE_WAIT_S))
  until idle; do [ $SECONDS -ge $deadline ] && { log "$1: card $CARD not idle after $KIT_IDLE_WAIT_S s; not run"; exit 3; }; sleep 30; done
}
cut_aside() { local s; s=$(date -u +%Y%m%dT%H%M%SZ); mv "$1" "$R/cut/$(basename "$1").$s"; log "$(basename "$1"): an earlier attempt did not finish; moved to cut/ and asked again"; }

# The placement is pinned for the whole Max-Q cell (PREREG.md; addendum F pinned it the same way on the second
# Workstation sitting). Stage 0 runs as a recorded reading in the timed slot.
export MEMRA_KV_VMM_GROW=inline
log "untimed half start card=$CARD lane=$(cat "$B/lane/source.commit") main=$(cat "$B/main/source.commit") harness=$(cat "$WT/SOURCE.commit") MEMRA_KV_VMM_GROW=inline"

# A2: the G1 grow series (run-grow.sh as chain.sh:81-86 ran it; the gate's tokenwise program, the collector's hold).
G=$R/grow-32768-maxq
if [ ! -e "$G/collector.exit" ]; then
  [ -d "$G" ] && cut_aside "$G"
  wait_idle "A2"
  env -u MEMRA_KV_VMM_GROW WT="$WT" ROOT="$R" GATE_BIN="$B/lane/kv-tier-gate" ARTIFACT="$MODEL" ATTEMPTS=2 RETRY_SECONDS=60 \
    bash research/spill-b-20260919/rtx5090-day37/run-grow.sh pro-single grow-32768-maxq > "$R/grow.log" 2>&1
  log "A2 rc=$? $(head -1 "$G/receipt/GROW.txt" 2>/dev/null)"
else
  log "A2 done already ($(head -1 "$G/receipt/GROW.txt" 2>/dev/null | cut -c1-60))"
fi

# A1: the gate set per arm, one collector hold each, the registered cell order (gates.sh).
export WT RIG_LOCK=/tmp/memra-gpu.lock MODEL MODEL_TWIN=$MODEL BIN="$B/lane/memra-server" HOSTGATE_MB=256
for arm in pooled vmm; do
  out=$R/gates-maxq-$arm
  if [ -e "$out/battery.log" ] && grep -q "gates done arm=$arm" "$out/battery.log"; then log "gates $arm done already"; continue; fi
  [ -d "$out" ] && cut_aside "$out"
  wait_idle "gates $arm"; mkdir -p "$out"
  python3 tools/tier-battery.py --rig pro-single --timeout 10800 --out "$out/collector" --external-lock --execute \
    bash research/spill-b-20260919/rtx5090-day37/gates.sh @COLLECTOR_LOCK_FD@ "$arm" "$out" > "$out/collector.log" 2>&1
  log "gates $arm rc=$?"
done

# The serving boots (boots-f.sh: the main arm clears every lane-only name, addendum F).
export LANE_BIN="$B/lane/memra-server" MAIN_BIN="$B/main/memra-server" MODEL_KEY=q38 BOOT_CTX='' STREAM_CONC=16 NO_SCOPE=1
bash research/spill-b-20260919/rtx5090-day37/boots-f.sh "$R" \
  mix-spec-O1-pooled:pooled:mixspec mix-spec-O1-vmm:vmm:mixspec mix-plain-O1-pooled:pooled:mixplain mix-plain-O1-vmm:vmm:mixplain \
  mix-spec-O2-vmm:vmm:mixspec mix-spec-O2-pooled:pooled:mixspec mix-plain-O2-vmm:vmm:mixplain mix-plain-O2-pooled:pooled:mixplain \
  off-lane:pooled:mixspec fault-mapper:vmm-mapperfault:mixspec off-main:main:mixspec \
  burst-g2-pooled:pooled:g2 burst-g2-vmm:vmm:g2 burst-l64-vmm:vmm:l64 burst-l64-pooled:pooled:l64 burst-boff-pooled:pooled:boff burst-boff-vmm:vmm:boff \
  fault-ensure:vmm-ensurefault:g2 fault-build1:vmm-buildfault1:g2 fault-build64:vmm-buildfault64:g2
rc=$?
log "boots rc=$rc"
[ $rc = 0 ] && log "MAXQ-KVVMM-UNTIMED-DONE"
exit $rc
