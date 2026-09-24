#!/usr/bin/env bash
# WP-A day 37 on the local RTX 5090 Laptop GPU (DAY37.md section 1): finding 5's reproduction protocol, fixed before it
# runs. ONE bounded hold of /tmp/memra-5090.lock (60 x 120 s behind the other lanes, never inside another hold; then no
# compute app and >= 20000 MiB free, bounded 15 x 60 s). Arms, in the hold: `pair` (the two finding-5 cells alone,
# --test-threads=2) N runs, then `all` (the eleven native cells of tier_transfer.rs in one process at the default
# thread count) N runs. Every run's output teed raw. 250 ms card telemetry. Executed-not-qualified.
# usage: stress.sh <out_dir> <engine test binary> <runs per arm> [<label>]
set -uo pipefail
OUT=$1; BIN=$2; N=$3; LABEL=${4:-repro}
HERE=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$HERE" || exit 1
LOCK=/tmp/memra-5090.lock
mkdir -p "$OUT/raw"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }
sha256sum "$BIN" > "$OUT/binary.sha256"
log "label=$LABEL tree=$(git rev-parse HEAD) binary=$(cut -c1-16 "$OUT/binary.sha256") runs_per_arm=$N"
exec 9>"$LOCK"
held=0
for attempt in $(seq 1 60); do
    if flock -w 120 9; then held=1; break; fi
    log "hold attempt $attempt: the lock stayed busy for 120 s"
done
[ "$held" = 1 ] || { log "NOT RUN: the lock never came free"; exit 2; }
log "hold taken (fd 9)"
idle=0
for attempt in $(seq 1 15); do
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$apps" ] && [ "${free_mib:-0}" -ge 20000 ]; then idle=1; break; fi
    log "idle wait $attempt under the hold: free=${free_mib}MiB apps=[${apps//$'\n'/; }]"
    sleep 60
done
[ "$idle" = 1 ] || { log "NOT RUN: the card never went idle under the hold"; flock -u 9; exit 2; }
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
    --format=csv -lms 250 > "$OUT/card-250ms.csv" 2>&1 &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT
PAIR=(tier_transfer::tests::d2h_span_batch_lands_with_its_ticket_on_the_copy_stream
      tier_transfer::tests::h2d_span_batch_lands_with_its_ticket_on_the_copy_stream)
for r in $(seq 1 "$N"); do
    "$BIN" --ignored --exact --test-threads=2 --nocapture "${PAIR[@]}" > "$OUT/raw/pair-$r.log" 2>&1
    log "pair run $r rc=$? $(grep -h '^test result' "$OUT/raw/pair-$r.log" | head -1)"
done
for r in $(seq 1 "$N"); do
    "$BIN" --ignored --nocapture tier_transfer::tests:: > "$OUT/raw/all-$r.log" 2>&1
    log "all run $r rc=$? $(grep -h '^test result' "$OUT/raw/all-$r.log" | head -1)"
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps.after.csv" 2>&1
log "hold released"
