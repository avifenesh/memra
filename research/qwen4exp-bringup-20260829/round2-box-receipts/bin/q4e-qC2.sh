#!/usr/bin/env bash
# agent C measurement queue, REVISION 2 (2026-08-31, after the card + lock corrections).
#
# Three rules this revision exists to obey, and why each one is not optional:
#  1. CUDA_VISIBLE_DEVICES=1. Two cards hold at most two lanes by construction (89,971 MiB
#     loaded, 95,805 peak at a filled 262k rung), so a three-way split over-subscribes. Card 0
#     is agent A's; I share card 1 with agent B by taking turns.
#  2. flock -x around the ENTIRE invocation - load, prefill AND timing. The in-instrument lock
#     covered timed ROUNDS only, so prefills ran unlocked and three lanes computed at once; that
#     invalidated three A/B attempts including my own 32,768 kvhoist row. Holding the lock across
#     the load also fixes the VRAM race for free: nobody allocates while somebody else holds it.
#  3. MEMRA_Q4E_MEASURE_LOCK stays UNSET. One mode per cell - a shell holding the lock whose child
#     instrument then asks for it again on the same path blocks on its own ancestor forever.
set -uo pipefail
Q=~/realgate/kvq2/QUEUE.log
BIN=~/realgate/bin/qwen4exp_real_gate.agentC
OUT=~/realgate/kvq2
CK=~/data/q48fn-yarn1m
IDS=~/realgate/ladder-ids.txt
LOCK=/tmp/q48fn-measure.lock
log(){ printf '[%s] qC: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$Q"; }
export CUDA_VISIBLE_DEVICES=1
export MEMRA_Q4E_SEAMS=idxsel   # the 262k target regime; measuring a lever without it prices it
                                # against a total inflated by a host section that is 83% of a
                                # prefill chunk at 131k, and dilutes the verdict toward "no effect".
unset MEMRA_Q4E_MEASURE_LOCK


# CAPACITY GUARD (agent A's suggestion, and a failure I was about to walk into). Winning the lock
# is NOT the same as having a card: the lock holder's process can release the lock and still be
# resident, and my cell would then take its turn and OOM on card 1 - burning a scarce turn and
# banking a confusing receipt instead of a number. So after acquiring the lock, wait for the card
# to actually be free, and if it never frees, say so LOUDLY and give the turn up rather than fail
# obscurely. Needs 91,000 MiB: a loaded model is 89,971 and the deepest rung here adds ~1.4 GiB.
need_card() {
  local need=91000 waited=0
  while [ $waited -lt 600 ]; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d " ")
    local free=$((97887 - ${used:-97887}))
    [ "$free" -ge "$need" ] && { echo "$waited"; return 0; }
    sleep 10; waited=$((waited+10))
  done
  return 1
}

export -f need_card
log "queueC-rev2 start card=1 lock=whole-invocation instrument=$(md5sum $BIN|cut -c1-16) src=$(cd ~/memra-c && git rev-parse --short HEAD)"

cell(){ local label=$1; shift
  log "cell $label WAITING for flock -x (whole invocation, incl. load+prefill)"
  local t0=$SECONDS
  flock -x "$LOCK" bash -c '
    set -uo pipefail
    if ! w=$(need_card); then
      echo "CAPACITY-GUARD: card 1 never freed 91000 MiB within 600 s - giving up this turn" >&2
      exit 90
    fi
    echo "# capacity-guard	card=1	waited_s=$w" 
    exec env CUDA_VISIBLE_DEVICES=1 MEMRA_Q4E_SEAMS=idxsel "$0" "$1" "$2" --label "$3" "${@:4}"
  ' "$BIN" "$CK" "$OUT" "$label" "$@" > "$OUT/$label.log" 2>&1
  local rc=$?
  log "cell $label rc=$rc total_s=$((SECONDS-t0)) verdict=$(grep -h ab-verdict "$OUT/$label.log" 2>/dev/null | tail -1 | tr '\t' ' ' | cut -c1-160)"
  return 0
}

# C1b: kvhoist, ONE shallow rung. Valid shallow because qsa.sdpa is depth-INVARIANT (banked flat
# at 10.3 ms across 100k/131k/150k; the selection budget saturates at ~2,052 rows from ~8k).
# ~10 min exclusive instead of ~25, which is the point of sizing it this way.
cell C1b-kvhoist --ladder 32768 --ladder-ids "$IDS" --ladder-chunk 2048 \
  --ladder-decode 36 --ladder-ab-seam kvhoist --ladder-ab-rounds 3 --ladder-ab-steps 16

# C2b: poolT, TWO rungs on ONE fill. The pooled score is O(n_blocks), so the falsifiable
# prediction is that the delta GROWS from 32,768 to 131,072. A flat delta kills the lever.
cell C2b-poolT --ladder 32768,131072 --ladder-ids "$IDS" --ladder-chunk 2048 \
  --ladder-decode 36 --ladder-ab-seam poolT --ladder-ab-rounds 3 --ladder-ab-steps 16

log "queueC-rev2 done"
