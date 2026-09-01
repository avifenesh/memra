#!/usr/bin/env bash
# downsel lane (mtp14) measurement cells: does filling the sel matvecs' idle lanes actually
# pay on a box card?
#
# NONE OF THESE HAVE RUN. The lane had no approved timing hardware (the cloud fleet closed account-wide,
# no provider approved), and the rig is exactness-only (LAW:rig-gpu-exactness-only). The
# shape was selected from interleaved rig RATIOS with a bit-identical control arm and the
# ceiling was priced from box nsys absolutes already in this repo — see
# spec/downsel/DOWNSEL.md sections 2 and 4. These cells replace both with box rows.
#
# WHAT IS BEING MEASURED. `qmatvec_nvfp4_modelopt_sel_f32_v3` and its gufuse twin partition
# the pair loop over all 32 lanes; at this artifact's geometry the DOWN launch has pairs=20
# (lanes 20-31 idle for the whole kernel) and the gate+up launch pairs=80 (a 3-vs-2 tail).
# The `selgroup` seam runs the same math over a SUB-WARP of g lanes with 4 rows per lane.
# Ceiling: 6.42% of the K=5 round (136.2 -> 145.5 tok/s). Rig-composed realization estimate:
# ~4.2% (~142 tok/s). Default is OFF; the flip rule is DOWNSEL.md section 6.
#
# FOUR RULES, none optional, each from a receipted failure:
#  1. flock -x around the ENTIRE invocation - load, prefill AND timing. An in-instrument lock
#     covering timed rounds only let prefills run unlocked and invalidated three A/B attempts
#     (q4e-qC2.sh). Holding across the load fixes the VRAM race for free.
#  2. The capacity+idle guard sits INSIDE the hold. Checking the card and THEN blocking on
#     flock is a race: the sibling queue refills the card while this cell waits, and the load
#     OOMs a minute into an exclusive hold (the vfuse lesson). nvidia-smi free is not driver
#     free.
#  3. MEMRA_Q4E_MEASURE_LOCK stays UNSET. One lock mode per cell - a shell holding the lock
#     whose child instrument asks for the same path blocks on its own ancestor forever.
#  4. Every receipt names the binary's own sha and the source commit, from the binary
#     (rebuild-attribution law: a receipt naming a binary that did not run is the trap).
#
# ARM IDENTITY (LAW:ab-arm-identity): the arms differ ONLY in MEMRA_Q4E_SEAMS, and the value
# actually in force is echoed into every receipt. `selgroup` is SHAPE-valued, so `off` and
# `auto` are the two arms; a pinned shape (`dn:4:4+gu:16:4`) is carried in the env per
# invocation and never through the boolean --*-ab-seam harness, which restores a pin as
# `auto` (documented at seam_state).
set -uo pipefail
BIN=${BIN:-$HOME/realgate/bin/qwen4exp_real_gate.downsel}
SHAPEBIN=${SHAPEBIN:-$HOME/realgate/bin/sel_shape_probe.downsel}
CK=${CK:-$HOME/data/q48fn-yarn1m}
OUT=${OUT:-$HOME/realgate/downsel}
SRC=${SRC:-$HOME/downsel-wt}
IDS=${IDS:-$HOME/realgate/ladder-ids.txt}
LOCK=${LOCK:-/tmp/q48fn-measure.lock}
CARD=${CARD:-0}
CARD_TOTAL_MIB=${CARD_TOTAL_MIB:-97887}
Q=$OUT/QUEUE.log
mkdir -p "$OUT"
log(){ printf '[%s] downsel: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$Q"; }

# The 262k target regime. Measuring a lever without idxsel prices it against a total inflated
# by a host section that is 83% of a prefill chunk at 131k, which dilutes every verdict
# toward "no effect" (q4e-qC2.sh).
BASE_SEAMS=${BASE_SEAMS:-idxsel}
unset MEMRA_Q4E_MEASURE_LOCK
export CUDA_VISIBLE_DEVICES=$CARD

src_sha(){ (cd "$SRC" 2>/dev/null && git log -1 --format=%H) || echo UNKNOWN; }
bin_sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || echo UNKNOWN; }

# need_card <need_mib>: wait for the card to be free AND idle, INSIDE the hold. Idle as well
# as free because these cells quote microsecond kernel times.
need_card() {
  local need=$1 waited=0
  while [ "$waited" -lt 900 ]; do
    local used util free
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null | tr -d ' ')
    free=$(( ${CARD_TOTAL_MIB} - ${used:-$CARD_TOTAL_MIB} ))
    if [ "$free" -ge "$need" ] && [ -n "$util" ] && [ "$util" -lt 5 ]; then
      echo "$waited"; return 0
    fi
    sleep 10; waited=$((waited+10))
  done
  return 1
}
export -f need_card

# receipt_head <logfile> <label> <seams> <binary>
receipt_head(){
  { printf '# cell\t%s\n' "$2"
    printf '# seams_env\tMEMRA_Q4E_SEAMS=%s\n' "$3"
    printf '# selgroup_arm\t%s\n' "${4:-n/a}"
    printf '# ckpt\t%s\n' "$CK"
    printf '# kv_quant\t%s\n' "${KVQ_NOTE:-default (owner K=q8_0 / V=q5_1, hardcoded per-family)}"
    printf '# idxq\t%s\n' "${IDXQ_NOTE:-default}"
    printf '# corpus_commit\t%s\n' "$(src_sha)"
    printf '# card\t%s\n' "$CUDA_VISIBLE_DEVICES"
  } >> "$1"
}

# ---------------------------------------------------------------------------------------
# CELL A -- the shape ladder on a box card. No checkpoint, ~1.3 GiB, ~30 s per pass, so it
# interleaves between any other lane's multi-hour cells instead of waiting for the queue.
# Replaces DOWNSEL.md section 4's rig ratios with box ones, at BOTH shapes the kernels serve.
# Read the `dn:32:4+gu:32:4` row first: it is the shipped program through the new kernel
# (bit-identical), so `section_vs_ctl` is the shape effect and `section_vs_off` is what
# shipping delivers.
# ---------------------------------------------------------------------------------------
cell_selshape(){
  local t=$1 pass=$2
  # Separate `local`s on purpose: bash does NOT see an earlier assignment in the SAME `local`
  # statement, so `local t=$1 label="...$t..."` silently interpolates an EMPTY t and the cell
  # writes its receipt under the wrong name (shellcheck SC2318).
  local label="selshape-t$t-pass$pass"
  local log_f="$OUT/$label.tsv"
  : > "$log_f"
  receipt_head "$log_f" "$label" "$BASE_SEAMS" "ladder (probe-internal arms)"
  printf '# bin_sha256\t%s\n# bin\t%s\n' "$(bin_sha "$SHAPEBIN")" "$SHAPEBIN" >> "$log_f"
  log "cell $label acquiring -x (t=$t reps=25)"
  flock -x "$LOCK" env MEMRA_SELSHAPE_T="$t" MEMRA_SELSHAPE_REPS=25 bash -c '
    if ! w=$(need_card 4000); then
      echo "# CAPACITY-GUARD: card never free+idle within 900s in-hold - turn given up" >&2
      exit 90
    fi
    printf "# capacity_guard\twaited_s=%s\n" "$w"
    exec "$1"
  ' _ "$SHAPEBIN" >> "$log_f" 2>&1
  log "cell $label rc=$? rows=$(grep -c '^[a-z]' "$log_f" 2>/dev/null)"
}

# ---------------------------------------------------------------------------------------
# CELL B -- the serving number. selgroup OFF vs ON on the REAL K=5 spec loop.
#
# Both arms inside ONE hold, and the arm ORDER FLIPS every rep (rep 0 runs OFF,ON; rep 1 runs
# ON,OFF; ...) so drift over the hold cannot accumulate onto one arm
# (TRAP:monotone-sweep-inflates-the-lever). Three holds, each reported with its own spread,
# never averaged across holds without it.
#
# --spec-sampled is NOT set here on purpose: this cell measures the KERNEL, and greedy is the
# instrument that makes the round comparable (greedy-is-the-instrument law). Per that same law
# the sampled twin is what a SERVING DECISION needs, so the default-flip rule (DOWNSEL.md
# section 6) requires the sampled arm too -- run this cell again with --spec-sampled before
# anyone touches a default.
# ---------------------------------------------------------------------------------------
cell_spec_ab(){
  local rep=$1
  local label="spec-ab-rep$rep"
  local order=("off" "auto"); [ $((rep % 2)) -eq 1 ] && order=("auto" "off")
  log "cell $label acquiring -x (arms ${order[0]} then ${order[1]}, ONE hold)"
  flock -x "$LOCK" bash -c '
    set -uo pipefail
    bin="$1"; ck="$2"; out="$3"; rep="$4"; base="$5"; a1="$6"; a2="$7"
    if ! w=$(need_card 91000); then
      echo "CAPACITY-GUARD: card never free+idle within 900s in-hold" >&2; exit 90
    fi
    for arm in "$a1" "$a2"; do
      seams="$base"; [ "$arm" = auto ] && seams="$base,selgroup"
      lf="$out/spec-ab-rep$rep-$arm.log"
      { printf "# capacity_guard\twaited_s=%s\n" "$w"
        printf "# arm\t%s\n# seams_env\tMEMRA_Q4E_SEAMS=%s\n" "$arm" "$seams"
      } > "$lf"
      env MEMRA_Q4E_SEAMS="$seams" "$bin" "$ck" "$out" \
        --label "spec-ab-rep$rep-$arm" --mtp --spec-k 5 --spec-ab 5x64 >> "$lf" 2>&1
      printf "# rc\t%s\n" "$?" >> "$lf"
    done
  ' _ "$BIN" "$CK" "$OUT" "$rep" "$BASE_SEAMS" "${order[0]}" "${order[1]}"
  log "cell $label rc=$?"
  for arm in off auto; do
    receipt_head "$OUT/spec-ab-rep$rep-$arm.log" "$label-$arm" \
      "$BASE_SEAMS$([ $arm = auto ] && echo ,selgroup)" "$arm"
    printf '# bin_sha256\t%s\n' "$(bin_sha "$BIN")" >> "$OUT/spec-ab-rep$rep-$arm.log"
  done
}

# ---------------------------------------------------------------------------------------
# CELL C -- the t=1 PLAIN DECODE surface, which is a second independent reason this lever
# exists: `moe.sel_grouped` is 2.6 ms and 6.9-8.6% of a deep decode token (PROFILE-12 section
# 3), and the same two kernels serve it.
#
# This is also the cell most likely to KILL the shape. AUTO grows rows-per-warp, so the down
# launch runs 80 blocks per slot where the shipped kernel runs 640 - at t=1 that is 800 warps
# instead of 6,400. The rig (82 SMs) showed no loss; 188 SMs is a different block-slot
# arithmetic, and warp packing on this exact launcher already measured NEGATIVE once on plain
# decode (14.38 -> 15.13 ms, mtp6). A regression here beyond spread fails flip-rule item 2.
#
# Uses the shared --ladder-ab-seam harness, which alternates arm order per rep and restores
# the entry arm (selgroup reports its boolean state for exactly that reason).
# ---------------------------------------------------------------------------------------
cell_decode_ab(){
  local label=decode-ab
  local log_f="$OUT/$label.log"
  log "cell $label acquiring -x"
  flock -x "$LOCK" bash -c '
    set -uo pipefail
    bin="$1"; ck="$2"; out="$3"; ids="$4"; base="$5"
    if ! w=$(need_card 91000); then
      echo "CAPACITY-GUARD: card never free+idle within 900s in-hold" >&2; exit 90
    fi
    printf "# capacity_guard\twaited_s=%s\n" "$w"
    exec env MEMRA_Q4E_SEAMS="$base" "$bin" "$ck" "$out" --label decode-ab \
      --ladder 32768 --ladder-ids "$ids" --ladder-chunk 2048 --ladder-decode 36 \
      --ladder-ab-seam selgroup --ladder-ab-rounds 3 --ladder-ab-steps 16
  ' _ "$BIN" "$CK" "$OUT" "$IDS" "$BASE_SEAMS" > "$log_f" 2>&1
  log "cell $label rc=$? verdict=$(grep -h ab-verdict "$log_f" 2>/dev/null | tail -1 | tr '\t' ' ' | cut -c1-160)"
  receipt_head "$log_f" "$label" "$BASE_SEAMS (+selgroup on the ON arm, harness-flipped)" "off vs auto"
  printf '# bin_sha256\t%s\n' "$(bin_sha "$BIN")" >> "$log_f"
}

# ---------------------------------------------------------------------------------------
# CELL D -- the 262k product rung. A fatter warp with an 8x smaller grid is the shape most
# likely to behave differently under a long-context admission load, and the product window is
# where a regression would actually cost money. Two rungs on ONE fill so the falsifiable
# prediction is readable: the sel section is per-slot, so the delta should be FLAT in depth.
# A delta that grows or vanishes with depth means something other than the shape moved.
# ---------------------------------------------------------------------------------------
cell_rung262k(){
  local label=rung262k
  local log_f="$OUT/$label.log"
  log "cell $label acquiring -x (LONG: full 262,144 prefill)"
  flock -x "$LOCK" bash -c '
    set -uo pipefail
    bin="$1"; ck="$2"; out="$3"; ids="$4"; base="$5"
    if ! w=$(need_card 91000); then
      echo "CAPACITY-GUARD: card never free+idle within 900s in-hold" >&2; exit 90
    fi
    printf "# capacity_guard\twaited_s=%s\n" "$w"
    exec env MEMRA_Q4E_SEAMS="$base" "$bin" "$ck" "$out" --label rung262k \
      --ladder 32768,262144 --ladder-ids "$ids" --ladder-chunk 2048 --ladder-decode 36 \
      --ladder-ab-seam selgroup --ladder-ab-rounds 3 --ladder-ab-steps 16
  ' _ "$BIN" "$CK" "$OUT" "$IDS" "$BASE_SEAMS" > "$log_f" 2>&1
  log "cell $label rc=$? verdict=$(grep -h ab-verdict "$log_f" 2>/dev/null | tail -1 | tr '\t' ' ' | cut -c1-160)"
  receipt_head "$log_f" "$label" "$BASE_SEAMS (+selgroup on the ON arm, harness-flipped)" "off vs auto"
  printf '# bin_sha256\t%s\n' "$(bin_sha "$BIN")" >> "$log_f"
}

# ---------------------------------------------------------------------------------------
# Build and install, as DISTINCT basenames. The basename matters: a gate's stop() pkills by
# basename, and renaming the comparison binary orphans a VRAM-holding server
# (gate-stop-pkill-basename-trap). Never install over a basename another lane parks on.
#
#   cd "$SRC"
#   cargo build --release -p memra-engine --bin sel_shape_probe --bin qwen4exp_real_gate
#   install -m755 target/release/sel_shape_probe      ~/realgate/bin/sel_shape_probe.downsel
#   install -m755 target/release/qwen4exp_real_gate   ~/realgate/bin/qwen4exp_real_gate.downsel
#   # then verify the installed binary carries this source (rebuild-after-checkout law):
#   git log -1 --format='%H %s'
#
# Order: A first. It is 30 s, needs no checkpoint, and if it does not reproduce the rig's
# direction on this card there is no reason to spend an exclusive hold on B/C/D at all.
# ---------------------------------------------------------------------------------------
case "${1:-all}" in
  selshape)  for t in 6 1; do for p in 1 2 3; do cell_selshape "$t" "$p"; done; done ;;
  spec-ab)   for r in 0 1 2; do cell_spec_ab "$r"; done ;;
  decode-ab) cell_decode_ab ;;
  rung262k)  cell_rung262k ;;
  all)
    for t in 6 1; do for p in 1 2 3; do cell_selshape "$t" "$p"; done; done
    for r in 0 1 2; do cell_spec_ab "$r"; done
    cell_decode_ab
    cell_rung262k
    ;;
  *) echo "usage: $0 [selshape|spec-ab|decode-ab|rung262k|all]" >&2; exit 2 ;;
esac
log "queue done"
