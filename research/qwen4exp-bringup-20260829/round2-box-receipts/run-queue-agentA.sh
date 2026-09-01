#!/usr/bin/env bash
# qwen4exp 262k lane, AGENT A queue v2. Two changes over v1, both forced by a measured
# protocol failure rather than by preference:
#
# 1. LOCK MODES. v1 wrapped timed cells in an exclusive flock and ran correctness cells
#    unlocked. At 20:52Z three agents were computing at once while my exclusive pair held the
#    lock, because the sibling lanes' in-instrument lock covers TIMED ROUNDS ONLY — every
#    lane's load and prefill ran unlocked and starved the host half of my timed decode. So:
#    every cell here is locked, and the mode says what it is. `-s` (shared) for untimed
#    correctness work, which may overlap freely with the other lanes' untimed work; `-x`
#    (exclusive) for anything whose number is quoted, which now also excludes their prefills.
#    Never nested: a shell holding -s whose child asks -x on the same path blocks on its own
#    ancestor forever, so MEMRA_Q4E_MEASURE_LOCK stays UNSET here.
#
# 2. ONE FILL PER A/B. The idxsel and graph A/Bs use the sibling lanes' `--ladder-ab-seam`,
#    which interleaves both arms on a SINGLE prefill at the rung. That drops the exclusive
#    window from 3 x 30 min to ~20 min once, and it is a better instrument as well: the two
#    arms share the literal same KV/indexer state, so a prefill-wall difference cannot leak
#    into the decode comparison at all.
set -u
cd "$HOME/memra"
BIN=$HOME/realgate/bin/qwen4exp_real_gate.agentA2
CKPT=$HOME/data/q48fn-yarn1m
OUT=$HOME/realgate/kvq2
TR=$HOME/realgate/traces
IDS=$HOME/realgate/ladder-ids.txt
SH=$HOME/realgate/shapes
LK=/tmp/q48fn-measure.lock
Q=$OUT/QUEUE.log
mkdir -p "$TR"
say(){ echo "[$(date -u +%FT%TZ)] qA: $*" >> "$Q"; }

# ---- CAPACITY GUARD. The lock serialises MEASUREMENT; it does not stop another lane from
# holding 90 GB of trunk on the card I need. qwen4_exp is 89,971 MiB post-load and a filled
# 262,144 rung peaks at 95,805 of 97,887, so a deep cell needs card 0 EMPTY — not "mostly
# free". Every cell below waits for that instead of racing into an OOM (three of my A/B
# attempts were already voided by contention; a fourth would be a choice, not an accident).
free_card0(){
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  [ -n "$used" ] && [ "$used" -lt 2000 ]
}
wait_card0(){
  local waited=0
  until free_card0; do
    if [ $((waited % 600)) -eq 0 ]; then
      say "WAITING for card 0 to clear before $1 (used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0 | tr -d ' '), waited ${waited}s)"
    fi
    sleep 30; waited=$((waited + 30))
  done
  [ "$waited" -gt 0 ] && say "card 0 clear after ${waited}s, starting $1"
  return 0
}

say "queueA2 start instrument=$(sha256sum "$BIN" | cut -c1-16) src=$(git -C "$HOME/memra" log -1 --format=%h) lockmodes=s/x"

# ---- 1. TIMED (-x): idxsel A/B on ONE 131,072 fill, both arms interleaved, x5 rounds.
# This is the POSITIVE CONTROL for --ladder-ab-seam as well as the A/B: idxsel at this depth
# has a known ~1.76x effect from the two-process cells, so a wash here would mean the seam
# toggle does not engage inside one run (captured decode graphs bake some kernel choices —
# the `selv2` seam note says so explicitly) rather than that the seam does not work.
wait_card0 "abseam-idxsel-131k"

wait_card0 "abseam-idxsel-131k"

say "cell abseam-idxsel-131k acquiring -x"
flock -x "$LK" -c "MEMRA_Q4E_SEAMS=idxsel $BIN $CKPT $OUT --label r2abseam-idxsel-131k \
  --ladder 131072 --ladder-ids $IDS --ladder-chunk 2048 --ladder-decode 36 \
  --ladder-ab-seam idxsel --ladder-ab-rounds 5 --ladder-ab-steps 32 \
  > $OUT/abseam-idxsel-131k.log 2>&1"
wait_card0 "abseam-idxsel-131k"

wait_card0 "abseam-idxsel-131k"

say "cell abseam-idxsel-131k rc=$?"

# ---- 2. TIMED (-x): the 262,144 headline rung, idxsel ON. Its claim INCLUDES the prefill
# wall (the 4,779 s -> ? number), so the whole cell is exclusive by necessity, not by habit.
wait_card0 "262k-on"

wait_card0 "262k-on"

say "cell 262k-on acquiring -x"
flock -x "$LK" -c "MEMRA_Q4E_SEAMS=idxsel $BIN $CKPT $OUT --label r2ship-262k-idxsel \
  --ladder 262144 --ladder-ids $IDS --ladder-chunk 2048 --ladder-decode 36 \
  > $OUT/ship-262k-idxsel.log 2>&1"
wait_card0 "262k-on"

wait_card0 "262k-on"

say "cell 262k-on rc=$?"

# ---- 3. TIMED (-x): GRAPHS AT DEPTH, one fill, arms interleaved. Round 4 measured decode
# graphs a wash and PROFILE-9 §3a found the setting stopped mattering once the devtwin stack
# removed the host boundaries (13.57 ON vs 13.60 OFF) — both SHALLOW, and both BEFORE idxsel.
# At 131k the per-token GPU work is 32.6 ms against 13.6 ms shallow, so the launch-issue share
# is smaller still and the prediction is "even more of a wash". Prediction is not receipt.
wait_card0 "abseam-graph-131k"

wait_card0 "abseam-graph-131k"

say "cell abseam-graph-131k acquiring -x"
flock -x "$LK" -c "MEMRA_Q4E_SEAMS=idxsel $BIN $CKPT $OUT --label r2abseam-graph-131k \
  --ladder 131072 --ladder-ids $IDS --ladder-chunk 2048 --ladder-decode 36 \
  --ladder-ab-seam graph --ladder-ab-rounds 5 --ladder-ab-steps 32 \
  > $OUT/abseam-graph-131k.log 2>&1"
wait_card0 "abseam-graph-131k"

wait_card0 "abseam-graph-131k"

say "cell abseam-graph-131k rc=$?"

# Same question at the target window, since the whole point is 262,144 and the launch/GPU
# ratio keeps moving with depth.
wait_card0 "abseam-graph-262k"

wait_card0 "abseam-graph-262k"

say "cell abseam-graph-262k acquiring -x"
flock -x "$LK" -c "MEMRA_Q4E_SEAMS=idxsel $BIN $CKPT $OUT --label r2abseam-graph-262k \
  --ladder 262144 --ladder-ids $IDS --ladder-chunk 2048 --ladder-decode 36 \
  --ladder-ab-seam graph --ladder-ab-rounds 5 --ladder-ab-steps 32 \
  > $OUT/abseam-graph-262k.log 2>&1"
wait_card0 "abseam-graph-262k"

wait_card0 "abseam-graph-262k"

say "cell abseam-graph-262k rc=$?"

# ---- 4. UNTIMED (-s): the depth gates. Oracle-free self-consistency, no number quoted.
for fill in 131072 262144; do
  wait_card0 "vbitdeep-$fill"

  say "cell vbitdeep-$fill (-s)"
  flock -s "$LK" -c "MEMRA_Q4E_SEAMS=idxsel $BIN $CKPT $OUT --label r2vbitdeep-$fill \
    --mtp --mtp-dev1 --spec-pmin 0.3 --spec-adapt 1 --spec-k 5 \
    --ladder-ids $IDS --ladder-chunk 2048 \
    --verify-bit-deep $fill --verify-bit-gate 24 > $OUT/vbitdeep-$fill.log 2>&1"
  wait_card0 "vbitdeep-$fill"

  say "cell vbitdeep-$fill rc=$?"
done

# Spec byte-identity AT DEPTH: a plain greedy chain (768 tokens) and a spec chain (256 at ship
# admission) at the same rung over the same raw corpus prefix. Both greedy from the identical
# fed sequence, so identity is "the first 256 banked continuation_ids agree" — same-config, no
# oracle. Untimed: the wall clocks here are not quoted anywhere.
wait_card0 "specid-plain-131k"

wait_card0 "specid-plain-131k"

say "cell specid-plain-131k (-s)"
flock -s "$LK" -c "MEMRA_Q4E_SEAMS=idxsel $BIN $CKPT $OUT --label r2specid-plain-131k \
  --ladder 131072 --ladder-ids $IDS --ladder-chunk 2048 --ladder-decode 768 \
  > $OUT/specid-plain-131k.log 2>&1"
wait_card0 "specid-plain-131k"

wait_card0 "specid-plain-131k"

say "cell specid-plain-131k rc=$?"
say "cell specid-spec-131k (-s)"
flock -s "$LK" -c "MEMRA_Q4E_SEAMS=idxsel $BIN $CKPT $OUT --label r2specid-spec-131k \
  --mtp --mtp-dev1 --spec-pmin 0.3 --spec-adapt 1 --spec-k 5 \
  --ladder 131072 --ladder-ids $IDS --ladder-chunk 2048 --ladder-decode 256 --ladder-spec 5 \
  > $OUT/specid-spec-131k.log 2>&1"
wait_card0 "specid-spec-131k"

wait_card0 "specid-spec-131k"

say "cell specid-spec-131k rc=$?"

# ---- 5. TIMED (-x): spec at depth, 262,144, per shape, ship admission. tok/s and accept
# rates are quoted, so these are exclusive.
for shape in thinkon thinkoff raw; do
  wait_card0 "spec262k-$shape"

  say "cell spec262k-$shape acquiring -x"
  if [ "$shape" = raw ]; then SHARG=""; else SHARG="--ladder-spec-shape $SH/$shape-prompts.tsv"; fi
  flock -x "$LK" -c "MEMRA_Q4E_SEAMS=idxsel $BIN $CKPT $OUT --label r2spec262k-$shape \
    --mtp --mtp-dev1 --spec-pmin 0.3 --spec-adapt 1 --spec-k 5 \
    --ladder 262144 --ladder-ids $IDS --ladder-chunk 2048 --ladder-decode 36 \
    --ladder-spec 5 $SHARG > $OUT/spec262k-$shape.log 2>&1"
  wait_card0 "spec262k-$shape"

  say "cell spec262k-$shape rc=$?"
done

# ---- 6. UNTIMED (-s): router traces, by shape and depth, for the co-activation placement
# lane (memra main tools/build_expert_placement_map.py via MEMRA_Q4E_EP_MAP). This lane does
# NOT build the placement map. The trace rides MEMRA_Q4E_ROUTER_AUDIT=1's host recompute, per
# the trace_moe_routes doc, because the shipped single-card route is device-side with no
# readback. Depth 32,768 deliberately: co-occurrence needs shapes, not the target window, and
# a traced 262k run writes ~700 MB per shape under an audit-slowed forward.
for shape in thinkon thinkoff raw; do
  wait_card0 "trace-$shape-32k"

  say "cell trace-$shape-32k (-s)"
  if [ "$shape" = raw ]; then SHARG=""; else SHARG="--ladder-spec-shape $SH/$shape-prompts.tsv"; fi
  flock -s "$LK" -c "MEMRA_Q4E_SEAMS=idxsel MEMRA_Q4E_ROUTER_AUDIT=1 \
    MEMRA_MOE_TRACE=$TR/moe-$shape-32768.trace \
    $BIN $CKPT $OUT --label r2trace-$shape-32k \
    --mtp --mtp-dev1 --spec-pmin 0.3 --spec-adapt 1 --spec-k 5 \
    --ladder 32768 --ladder-ids $IDS --ladder-chunk 2048 --ladder-decode 60 \
    --ladder-spec 5 $SHARG > $OUT/trace-$shape-32k.log 2>&1"
  say "cell trace-$shape-32k rc=$? bytes=$(stat -c%s "$TR/moe-$shape-32768.trace" 2>/dev/null)"
  gzip -f "$TR/moe-$shape-32768.trace" 2>/dev/null
done
say "queueA2 done"
