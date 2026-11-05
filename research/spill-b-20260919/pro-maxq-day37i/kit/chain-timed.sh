#!/usr/bin/env bash
# kv-vmm Max-Q cell, the timed half, for a quiet box (no other compute on any card): stage 0 as a recorded reading
# (vmm-call-cost, chain.sh:69-74's command), then the A3 stream pairs, stream-O1-k (pooled then vmm) and stream-O2-k
# (vmm then pooled) for k = 1..5, in chain.sh:106-109's order (A3 (i), A3 (ii) and A1-STREAM). Every boot through
# boots-f.sh with the box-wide idle check (KIT_IDLE_SCOPE=box, the registered form).
# Run only by run-timed.sh, into the same OUT as the untimed half so the reader reads one receipt root.
set -uo pipefail
: "${OUT:?}" "${K:?}" "${CARD:?}" "${KIT_HARNESS:?}" "${KIT_LOCK_FD:?}" "${MODEL:?}"
R=$OUT; WT=$KIT_HARNESS; B=$K/bins
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
idle() {
  python3 tools/tier-lock-proof.py --fd "$KIT_LOCK_FD" --lock /tmp/memra-gpu.lock > /dev/null 2>&1 || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
}
wait_idle() {
  local deadline=$((SECONDS + KIT_IDLE_WAIT_S))
  until idle; do [ $SECONDS -ge $deadline ] && { log "$1: box not idle after $KIT_IDLE_WAIT_S s; not run"; exit 3; }; sleep 30; done
}
log "timed half start card=$CARD MEMRA_KV_VMM_GROW=inline"

# Stage 0: a reading. The placement stays pinned to inline whatever it prints (PREREG.md).
if ! grep -q '^GROW-PLACEMENT' "$R/stage0/probe.log" 2>/dev/null; then
  if [ -d "$R/stage0" ]; then s=$(date -u +%Y%m%dT%H%M%SZ); mv "$R/stage0" "$R/cut/stage0.$s"; log "stage0: an earlier attempt did not finish; moved to cut/"; fi
  wait_idle "stage 0"; mkdir -p "$R/stage0"
  python3 tools/tier-battery.py --rig pro-single --timeout 1800 --out "$R/stage0/collector" --execute \
    bash -c "'$B/lane/vmm-call-cost' --reps 20 --queue-ms 60 --extents 1,4,16,64 | tee '$R/stage0/probe.log'" > "$R/stage0/collector.log" 2>&1
  log "stage 0 (a reading) rc=$? $(grep -h '^GROW-PLACEMENT\|^RELEASE-PLACEMENT' "$R/stage0/probe.log" | tr '\n' ' ')"
else
  log "stage 0 done already: $(grep -h '^GROW-PLACEMENT' "$R/stage0/probe.log")"
fi
echo inline > "$R/stage0/PLACEMENT"

# A stream pair is one unit (DAY37 addendum H): a pair a deadline cut is asked again whole, inside one window.
for o in O1 O2; do for k in 1 2 3 4 5; do
  p=$R/boots/stream-$o-$k-pooled; v=$R/boots/stream-$o-$k-vmm
  if [ -e "$p.exit" ] && [ -e "$v.exit" ]; then continue; fi
  if [ -e "$p.arm.txt" ] || [ -e "$v.arm.txt" ]; then
    s=$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$R/boots/cut"
    for x in "$p" "$p.arm.txt" "$p.launch.log" "$p.exit" "$v" "$v.arm.txt" "$v.launch.log" "$v.exit"; do
      [ -e "$x" ] && mv "$x" "$R/boots/cut/$(basename "$x").$s"
    done
    log "stream-$o-$k: pair cut by an earlier window; both boots moved to boots/cut/*.$s and the pair asked again"
  fi
done; done

export MEMRA_KV_VMM_GROW=inline KIT_IDLE_SCOPE=box
export WT RIG_LOCK=/tmp/memra-gpu.lock MODEL
export LANE_BIN="$B/lane/memra-server" MAIN_BIN="$B/main/memra-server" MODEL_KEY=q38 BOOT_CTX='' STREAM_CONC=16 NO_SCOPE=1
args=()
for k in 1 2 3 4 5; do args+=("stream-O1-$k-pooled:pooled:stream" "stream-O1-$k-vmm:vmm:stream"); done
for k in 1 2 3 4 5; do args+=("stream-O2-$k-vmm:vmm:stream" "stream-O2-$k-pooled:pooled:stream"); done
bash research/spill-b-20260919/rtx5090-day37/boots-f.sh "$R" "${args[@]}"
rc=$?
log "stream boots rc=$rc"
[ $rc = 0 ] && log "MAXQ-KVVMM-TIMED-DONE"
exit $rc
