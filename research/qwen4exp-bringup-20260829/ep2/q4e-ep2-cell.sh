#!/usr/bin/env bash
# EP2 lane box cells for qwen4_exp (Qwen3.8-Flash-Next-NVFP4).
#
# NONE OF THESE HAVE RUN. The lane had no timing hardware: the rig is one sm_120a laptop
# 5090, exactness-only (LAW:rig-gpu-exactness-only) and physically cannot run a two-card
# cell. The verdict in EP2-DESIGN.md was reached from box receipts already banked in this
# repo, not from these cells, and none of them is needed to reach it.
#
# READ EP2-DESIGN.md FIRST. Its verdict is that TWO-CARD EP2 CANNOT REACH THE 200 tok/s
# TARGET: from the measured 136.2 tok/s / 31.94 ms K=5 round, 200 needs 10.19 ms out, a
# two-card split of a fraction f of the round saves at most f/2, so it needs f >= 63.8%
# routed-expert work, and every attribution in the repo puts f between 22.9% and 48.5%.
# EP2's two-card ceiling is 154-180 tok/s. So DO NOT spend a box turn on cells B-E to chase
# the target. Cell A is the one worth running on its own merits (it settles a 2x attribution
# spread that every future MoE lever is sized against, in one short hold). Cells B-D exist so
# that IF two cards get committed for another reason, the +13-32% is one queue away instead of
# one lane away.
#
# FOUR RULES, none optional, each from a receipted failure in this lane (copied from
# spec/downsel/q4e-downsel-cell.sh, which learned them):
#  1. flock -x around the ENTIRE invocation - load, prefill AND timing. An in-instrument lock
#     covering timed rounds only let prefills run unlocked and invalidated three A/B attempts
#     (q4e-qC2.sh). Holding across the load fixes the VRAM race for free.
#  2. The capacity+idle guard sits INSIDE the hold. Checking the card and THEN blocking on
#     flock is a race: the sibling queue refills the card while this cell waits and the load
#     OOMs a minute into an exclusive hold (the vfuse lesson). nvidia-smi free is not driver
#     free.
#  3. MEMRA_Q4E_MEASURE_LOCK stays UNSET. One lock mode per cell - a shell holding the lock
#     whose child instrument asks for the same path blocks on its own ancestor forever.
#  4. Every receipt names the binary's own sha256 AND the source commit read FROM the binary's
#     tree (rebuild-attribution law: a receipt naming a binary that did not run is the trap),
#     plus its cache arm (`# cache kv_quant= idxq= golden_pin= seams_env=`) - this lane
#     shipped a silently f32-only instrument once (PROFILE-10 section 4).
#
# TWO-CARD CELLS NEED BOTH CARDS, so the capacity guard here is a PAIR guard. And the arm
# identity rule (LAW:ab-arm-identity) bites harder than usual: `--tp2` changes which BINARY
# PATH runs, not a flag inside one, so every two-card receipt must carry the
# `# expert-split ... engaged=true` line. A TP2 arm whose peer_slots delta is 0 measured a
# program that did not cross a card, and health-200 style "it ran" is not evidence.
set -uo pipefail
BIN=${BIN:-$HOME/realgate/bin/qwen4exp_real_gate}
CK=${CK:-$HOME/data/q48fn-yarn1m}
OUT=${OUT:-$HOME/realgate/ep2}
SRC=${SRC:-$HOME/ep2-wt}
LOCK=${LOCK:-/tmp/q48fn-measure.lock}
# EXPORTED, and that is not cosmetic. `need_cards` is exported with `export -f` and called
# inside every `flock -x ... bash -c` child, each of which runs `set -u`; an unexported
# CARD_TOTAL_MIB makes the guard die on `unbound variable` in the child and every cell then
# exits 90 with a message BLAMING THE CARD for a turn that never ran. Found by review before
# this script ever ran, reproduced in the shipped child shape, and both guard arms are now
# executed in that shape rather than in the parent (where the variable is always set and the
# bug is invisible). This is the loud-failures-fail-quietly law applied to the guard itself.
export CARD_TOTAL_MIB=${CARD_TOTAL_MIB:-97887}
EP_MAP=${EP_MAP:-$OUT/ep-map-coactivation.json}
TRACE_DIR=${TRACE_DIR:-$HOME/realgate/traces}
# The gate's SECOND POSITIONAL is its out dir; `--prompts` selects the shape pack. Packs are
# minted by spec/make-shape-prompts.py into {thinkon,thinkoff,efflow}-prompts.tsv; `raw` is the
# mtp2..mtp8 pack every perf row in this lane shares (acceptance 0.840 against the shapes'
# 0.290-0.588 - never quote one for the other, mtp9).
GATE_OUT=${GATE_OUT:-$OUT/gate}
PROMPTS_DIR=${PROMPTS_DIR:-$HOME/realgate/prompts}
RAW_PROMPTS=${RAW_PROMPTS:-$PROMPTS_DIR/raw-prompts.tsv}
Q=$OUT/QUEUE.log
mkdir -p "$OUT" "$TRACE_DIR" "$GATE_OUT"
log(){ printf '[%s] ep2: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$Q"; }

# The 262k target regime. Measuring anything without idxsel prices it against a total
# inflated by a host section that is 83% of a prefill chunk at 131k, which dilutes every
# verdict toward "no effect" (q4e-qC2.sh).
BASE_SEAMS=${BASE_SEAMS:-idxsel}
unset MEMRA_Q4E_MEASURE_LOCK

src_sha(){ (cd "$SRC" 2>/dev/null && git log -1 --format=%H) || echo UNKNOWN; }
bin_sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || echo UNKNOWN; }

# NEED_CARDS_WAIT_S exists ONLY so both arms of the guard can be executed in seconds. At the
# shipped 900 the red arm costs 15 minutes, and a guard whose failing arm is expensive to run is
# a guard nobody runs, which is how this file shipped a broken one past its own header once.
export NEED_CARDS_WAIT_S=${NEED_CARDS_WAIT_S:-900}

# need_cards <need_mib> <card...>: wait for EVERY named card to be free AND idle, INSIDE the
# hold. Idle as well as free because these cells quote per-round wall clocks. Exits loudly
# rather than obscurely: a guard proven only on its red arm may be unconditionally red, which
# is the same defect as unconditionally green, so BOTH arms are executed in the shipped child
# context (see the CARD_TOTAL_MIB note above) rather than in the parent shell.
#
# EXECUTED RECEIPT (rig, one card, NEED_CARDS_WAIT_S=20, inside `flock -x ... bash -c` + set -u):
#   green  need=100     -> rc 0,  prints waited=0
#   red    need=999999  -> rc 90, prints the capacity-guard line
# Before the CARD_TOTAL_MIB export both arms returned rc 90: unconditionally red, i.e. exactly
# as broken as unconditionally green and indistinguishable from a busy card in the receipt.
need_cards() {
  local need=$1; shift
  local cards=("$@") waited=0
  while [ "$waited" -lt "$NEED_CARDS_WAIT_S" ]; do
    local ok=1
    for c in "${cards[@]}"; do
      local used util free
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$c" 2>/dev/null | tr -d ' ')
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$c" 2>/dev/null | tr -d ' ')
      free=$(( CARD_TOTAL_MIB - ${used:-$CARD_TOTAL_MIB} ))
      if [ "$free" -lt "$need" ] || [ -z "$util" ] || [ "$util" -ge 5 ]; then ok=0; break; fi
    done
    [ "$ok" -eq 1 ] && { echo "$waited"; return 0; }
    sleep 10; waited=$((waited+10))
  done
  return 1
}
export -f need_cards

# receipt_head <logfile> <label> <seams> <cards> [extra note]
receipt_head(){
  { printf '# cell\t%s\n' "$2"
    printf '# seams_env\tMEMRA_Q4E_SEAMS=%s\n' "$3"
    printf '# cards\t%s\n' "$4"
    printf '# note\t%s\n' "${5:-}"
    printf '# ckpt\t%s\n' "$CK"
    printf '# cache\tkv_quant=%s\tidxq=%s\tgolden_pin=%s\tseams_env=%s\n' \
      "${KVQ_NOTE:-ship default (K=q8_0 / V=q5_1)}" "${IDXQ_NOTE:-ship default q8}" \
      "${PIN_NOTE:-false}" "$3"
    printf '# ep_map\t%s\n' "${EP_MAP_NOTE:-unset = EVEN split (control arm)}"
    printf '# bin\t%s\n# bin_sha256\t%s\n# src_commit\t%s\n' \
      "$BIN" "$(bin_sha "$BIN")" "$(src_sha)"
  } >> "$1"
}

# =======================================================================================
# CELL A -- THE ONE WORTH RUNNING. Single card, one short hold, no code, no new instrument.
#
# Settles a 2x spread in the same quantity on the same model. The routed-expert share of the
# K=5 round reads 22.9% (MOEUNION), 26.4% (DOWNSEL, from the mtp10 nsys kernel medians) and
# 48.5% (spec/mtp{4,5,6}/spec-profile-k5-*.tsv). The 48.5% rows are from a 66 tok/s program
# (963-982 ms for 64 tokens vs today's 7.341 ms/token) whose per-expert grouped executor was
# 30% of attributed on 177.2 calls/round -- an executor the merged verify path
# (`t > 1 && verify_mt_on() && sel_gufuse_on()`, 2 sel launches + t combines per layer) has
# since replaced. NOBODY RE-RAN IT after the program doubled. This is that re-run.
#
# FALSIFIABLE PREDICTION (so the cell can fail): the routed sections
# (moe.sel_grouped + moe.sel_bf16 + mtp.moe + surviving moe.dequant/moe.expert_gemms) come
# out at 25-35% of attributed with the per-expert terms collapsed from 30% toward zero,
# reconciling with the nsys 26.4%. Above 63.8% and the EP2 verdict is WRONG and the lane
# reopens; 35-63% keeps the verdict and moves the ceiling inside the 154-180 band.
#
# DEPTH IS PART OF THE RESULT, not a detail: qsa.sdpa is 1.2% in the shallow mtp6 profile and
# 29.7% in PROFILE-C0's 100k decode, because the QSA selection budget (512 blocks x 4 = 2,048
# tokens) SATURATES at ~2,052 rows from roughly 8k of fill onward. So this cell is SHALLOW and
# says so in its own receipt, which is fine for its purpose (it is the like-for-like re-run of
# the shallow mtp4/5/6 rows) and NOT fine for anything else.
#
# AND THE DEEP TWIN CANNOT BE SCRIPTED, so it is named instead of faked. `--spec-profile` runs
# a fixed 64-token spec loop on the prompt pack at the top of main, BEFORE the `--ladder`
# block; passing `--ladder 32768 --spec-profile 5` therefore emits a SHALLOW profile and then a
# separate ladder, silently answering a different question than the flags read like. There is
# `--ladder-spec` and `--ladder-spec-shape` but no `--ladder-spec-profile`: no instrument in
# this repo section-profiles a spec round at depth. That is a small instrument change (run the
# existing prof_section timers over the rung's spec loop instead of the shallow one) and it is
# the prerequisite for any deep MoE-share number. Until it lands, every routed-expert share in
# this lane -- including the verdict's -- is a SHALLOW-to-mid figure, which is stated in
# EP2-DESIGN.md rather than smoothed over.
#
# `--profile` refuses `--tp2`, so this cell is single-card by construction.
# =======================================================================================
cell_spec_profile(){
  local label="A-spec-profile-k5-shallow"
  local log_f="$OUT/$label.tsv"
  : > "$log_f"
  receipt_head "$log_f" "$label" "$BASE_SEAMS" "0" \
    "SHALLOW (64-token spec loop, QSA selection UNSATURATED); settles the 22.9/26.4/48.5% routed-expert spread against the mtp4/5/6 rows; no deep twin exists (no --ladder-spec-profile)"
  log "cell $label acquiring -x"
  flock -x "$LOCK" bash -c '
    set -uo pipefail
    if ! w=$(need_cards 91000 0); then
      echo "# CAPACITY-GUARD: card 0 never free+idle within 900s in-hold - turn given up" >&2
      exit 90
    fi
    printf "# capacity_guard\twaited_s=%s\n" "$w"
    bin=$1; ck=$2; go=$3; pr=$4; seams=$5; shift 5
    CUDA_VISIBLE_DEVICES=0 MEMRA_Q4E_SEAMS="$seams" \
      exec "$bin" "$ck" "$go" --label ep2A --prompts "$pr" --spec-profile 5 "$@"
  ' _ "$BIN" "$CK" "$GATE_OUT" "$RAW_PROMPTS" "$BASE_SEAMS" >> "$log_f" 2>&1
  log "cell $label rc=$?"
}

# =======================================================================================
# CELL B -- the two-card EXACTNESS gate across the EP seam. Correctness only, no timing, so
# it does not need an idle pair, only a free one -- but it takes the same hold because a
# neighbour computing on card 1 is exactly what a peer-join gate must not race.
#
# Three arms, and the third is the one this lane adds:
#   B1  --tp2-gate 24            decode logits vs the single-card twin. Established bar:
#                                24/24 argmax, worst_rel 3.016e-5, and it has reproduced
#                                BYTE-identically across two physical boxes (BASELINE.md).
#   B2  --tp2-prefill-gate 8     the calibrated two-regime class gate. Bands are MEASURED on
#                                this model, never borrowed: prime 1.4e-4 / decode 1.6e-4,
#                                red floor 1e-3. `decode_byte_identical=false` is a REPORTED
#                                FIELD here, not a failure: this program is an expert-half
#                                split with a partial-sum ADD join, not glm5's
#                                column-parallel-over-gather, so t=1 is a near-tie BAND class
#                                by construction. Do not "fix" it to byte identity.
#   B3  the three red arms       MEMRA_Q4E_TP2_GATE_RED=skip-peer-moe|peer-local-ids|
#                                reverse-peer-weights. A red arm PASSES BY BEING LOUD. Known
#                                magnitudes: 9.930e0 / 1.003e1 / 8.271e0, i.e. ~6-7e5x the
#                                green worst, argmax 13/19, 11/19, 13/19. peer-local-ids is
#                                both the most plausible real bug and the most damaging by
#                                argmax - the arm to keep if one is ever dropped.
#
# WHAT THIS LANE ADDS TO B: run every arm TWICE, once with EP_MAP unset (even split, the
# control) and once with a measured map. The map changes bytes MOVED, never bytes COMPUTED
# (ownership only selects which card runs the identical per-expert program; the combine stays
# slot-ordered), so B under a measured map must land in the SAME bands as B under the even
# split. That is the only correctness claim a placement A/B is allowed to rest on, and it has
# never been run because no map has ever been minted for this artifact (cell D).
#
# THE SPEC BYTE-IDENTITY ARM ASKED FOR BY THIS LANE'S BRIEF DOES NOT EXIST AND CANNOT BE
# WRITTEN HERE: `decode_step_tp2` is t==1-wired, there is no TP2 verify path, and the gate
# refuses `--ladder-tp2` with `--ladder-spec` ("spec at depth is single-card"). A two-card
# spec arm is ENGINE WORK (a t-generic TP2 verify plus a device-side route split, since the
# TP2 router is a host top-k with a per-MoE-layer [t,512] dtoh while the single-card serving
# path keeps the route on the card under `routerdev`). Cost that before promising it.
# =======================================================================================
cell_exactness(){
  local arm=$1                        # even | mapped
  local label="B-exactness-$arm"
  local log_f="$OUT/$label.tsv"
  local map=""
  if [ "$arm" = mapped ]; then
    [ -r "$EP_MAP" ] || { log "cell $label SKIPPED: no map at $EP_MAP (run cell D first)"; return 3; }
    map=$EP_MAP
  fi
  : > "$log_f"
  EP_MAP_NOTE=${map:-unset = EVEN split (control arm)} \
    receipt_head "$log_f" "$label" "$BASE_SEAMS" "0,1" \
      "correctness only; every row must carry engaged=true or it measured one card"
  log "cell $label acquiring -x (both cards)"
  flock -x "$LOCK" bash -c '
    set -uo pipefail
    bin=$1; ck=$2; go=$3; seams=$4; map=$5
    if ! w=$(need_cards 95000 0 1); then
      echo "# CAPACITY-GUARD: the PAIR was never free+idle within 900s in-hold" >&2; exit 90
    fi
    printf "# capacity_guard\twaited_s=%s\n" "$w"
    ep=(); [ -n "$map" ] && ep=(env "MEMRA_Q4E_EP_MAP=$map")
    for g in "--tp2-gate 24" "--tp2-prefill-gate 8"; do
      printf "\n# arm\tgreen %s\n" "$g"
      # shellcheck disable=SC2086
      "${ep[@]}" env MEMRA_Q4E_SEAMS="$seams" "$bin" "$ck" "$go" --label ep2B --tp2 $g || \
        printf "# arm-rc\t%s\n" "$?"
    done
    for red in skip-peer-moe peer-local-ids reverse-peer-weights; do
      printf "\n# arm\tRED %s (passes by being LOUD: expect ~6-7e5x the green worst)\n" "$red"
      "${ep[@]}" env MEMRA_Q4E_SEAMS="$seams" MEMRA_Q4E_TP2_GATE_RED="$red" \
        "$bin" "$ck" "$go" --label "ep2B-$red" --tp2 --tp2-prefill-gate 8 || \
        printf "# arm-rc\t%s\n" "$?"
    done
  ' _ "$BIN" "$CK" "$GATE_OUT" "$BASE_SEAMS" "$map" >> "$log_f" 2>&1
  log "cell $label rc=$?"
}

# =======================================================================================
# CELL C -- the t=1 plain decode A/B. This is the ONLY timed two-card arm the engine can run
# today, and it is what PROFILE-10 lists as owed ("the placement A/B is the NEXT lane").
#
# Three arms in ONE hold, arm order FLIPPED every rep so drift over the hold cannot
# accumulate onto one arm (LAW:interleaved-ab-protocol; the monotone sweep is what invalidated
# passes 1 and 2 of the moeu ladder):
#   single card | TP2 + even split (control) | TP2 + measured map
#
# The comparison that matters is arm 3 vs arm 2, NOT vs arm 1. Arm 1 is context: the
# established figures are single 15.57 ms vs TP2 14.22 ms (1.095x, perf/ab-tp2graphs2-nvfp4.tsv)
# and later single 14.5 vs TP2 12.6-12.9 (PROFILE-5). Every one of those lines is prefixed
# in-tree "# timing lines are UNTUNED EAGER wall clocks under correctness-arm residency - NOT
# perf claims" and carrying that scope forward is not optional.
#
# WHAT THIS CELL CAN AND CANNOT DECIDE. It can decide whether co-activation placement beats
# the even split on the shape the engine can run. It CANNOT justify any serving default:
# plain decode is 1.55x SLOWER than the single-card spec path we actually serve
# (single-card spec K=5 = 8.37 ms/token / 119.50 tok/s vs plain TP2's ~12.9), so a win here
# is a win on a shape no customer gets. Per the never-serve-greedy law a serving decision
# also needs the vendor-default SAMPLED rows and the 8-turn larger-prompt cache-on twin;
# neither is in this cell because there is no two-card serving arm to decide about yet.
# =======================================================================================
cell_decode_ab(){
  local rep=$1
  local label="C-decode-ab-rep$rep"
  local log_f="$OUT/$label.tsv"
  [ -r "$EP_MAP" ] || { log "cell $label SKIPPED: no map at $EP_MAP (run cell D first)"; return 3; }
  local order=("single" "even" "mapped")
  [ $((rep % 3)) -eq 1 ] && order=("even" "mapped" "single")
  [ $((rep % 3)) -eq 2 ] && order=("mapped" "single" "even")
  : > "$log_f"
  receipt_head "$log_f" "$label" "$BASE_SEAMS" "0,1" \
    "arms: ${order[*]} (order rotated per rep); compare mapped vs even, never vs single"
  log "cell $label acquiring -x (arms ${order[*]}, ONE hold)"
  flock -x "$LOCK" bash -c '
    set -uo pipefail
    bin=$1; ck=$2; go=$3; pr=$4; seams=$5; map=$6; rep=$7; shift 7
    if ! w=$(need_cards 95000 0 1); then
      echo "# CAPACITY-GUARD: the PAIR was never free+idle within 900s in-hold" >&2; exit 90
    fi
    printf "# capacity_guard\twaited_s=%s\n" "$w"
    for arm in "$@"; do
      printf "\n# arm\t%s\n" "$arm"
      case $arm in
        single) env MEMRA_Q4E_SEAMS="$seams" "$bin" "$ck" "$go" \
                  --label "ep2C-r$rep-single" --prompts "$pr" --decode-timing 40 ;;
        even)   env MEMRA_Q4E_SEAMS="$seams" "$bin" "$ck" "$go" --tp2 \
                  --label "ep2C-r$rep-even" --prompts "$pr" --decode-timing 40 ;;
        mapped) env MEMRA_Q4E_SEAMS="$seams" MEMRA_Q4E_EP_MAP="$map" "$bin" "$ck" "$go" --tp2 \
                  --label "ep2C-r$rep-mapped" --prompts "$pr" --decode-timing 40 ;;
      esac || printf "# arm-rc\t%s\n" "$?"
    done
  ' _ "$BIN" "$CK" "$GATE_OUT" "$RAW_PROMPTS" "$BASE_SEAMS" "$EP_MAP" "$rep" "${order[@]}" >> "$log_f" 2>&1
  log "cell $label rc=$?"
}

# =======================================================================================
# CELL D -- mint the placement map. THIS IS A PREREQUISITE FOR B(mapped) AND C, and it is
# still owed from round 2 ("work item 5, router traces: not collected").
#
# The gap that silently blanks this input, found while pricing the expert-speculation lever
# and worth restating because it makes an empty traces/ dir look like a config mistake:
# `trace_moe_routes` is called ONLY from the TP2 paths. The single-card device-routed default
# emits NOTHING - `route_topk_device` does the ROUTER_AUDIT readback and never calls the
# tracer - and the single-card host-routed `moe_forward` has no call either. So traces must be
# collected under --tp2, or with the tap wired onto the audit readback first.
#
# Collect all three shapes: the co-activation structure is a property of the traffic, and
# thinking-mode routing is not raw-mode routing. --spec-k 5 so the t=6 lines (60 ids,
# token-major) are present: spec/moeu/moe-union.py reads exactly that half of the trace and it
# is the union input the moeu lane never got.
#
# BALANCE IS A HARD CONSTRAINT, NOT A PREFERENCE: the card-1 bank halves are equal-size device
# allocations, so the engine REFUSES any layer where card 1 does not own exactly experts/2
# (256), naming --balance-tolerance. Mint balanced or the map will not load. Prior evidence
# says a coactivation map has real work to do: under the even split PROFILE-10 measured
# both_card_fraction=0.9993 with peer_slot_fraction=0.5140, i.e. essentially every layer-token
# pays a cross-card join today.
# =======================================================================================
cell_mint_map(){
  local label="D-mint-ep-map"
  local log_f="$OUT/$label.tsv"
  : > "$log_f"
  receipt_head "$log_f" "$label" "$BASE_SEAMS" "0,1" \
    "collects MEMRA_MOE_TRACE under --tp2 (the single-card default emits nothing) and mints the map"
  log "cell $label acquiring -x (both cards)"
  flock -x "$LOCK" bash -c '
    set -uo pipefail
    bin=$1; ck=$2; go=$3; pd=$4; seams=$5; td=$6
    if ! w=$(need_cards 95000 0 1); then
      echo "# CAPACITY-GUARD: the PAIR was never free+idle within 900s in-hold" >&2; exit 90
    fi
    printf "# capacity_guard\twaited_s=%s\n" "$w"
    for shape in thinkon thinkoff efflow; do
      tf="$td/moe-$shape.trace"
      printf "\n# trace-shape\t%s\n# trace_file\t%s\n" "$shape" "$tf"
      env MEMRA_Q4E_SEAMS="$seams" MEMRA_Q4E_ROUTER_AUDIT=1 MEMRA_MOE_TRACE="$tf" \
        "$bin" "$ck" "$go" --label "ep2D-$shape" --prompts "$pd/$shape-prompts.tsv" \
          --tp2 --spec-gate 256 --spec-k 5 || printf "# arm-rc\t%s\n" "$?"
      # A trace tap that emitted NOTHING is the failure this counter exists to catch: the
      # single-card default never calls trace_moe_routes, so an empty file here means the
      # --tp2 arm did not run, not that the model routed nothing.
      printf "# trace_lines\t%s\n" "$(wc -l < "$tf" 2>/dev/null || echo 0)"
    done
  ' _ "$BIN" "$CK" "$GATE_OUT" "$PROMPTS_DIR" "$BASE_SEAMS" "$TRACE_DIR" >> "$log_f" 2>&1
  # Mint OUTSIDE the hold: it is host-only work and must not sit on a card.
  {
    printf '\n# mint\ttools/build_expert_placement_map.py\n'
    python3 "$SRC/tools/build_expert_placement_map.py" \
      --trace "$TRACE_DIR"/moe-thinkon.trace \
      --trace "$TRACE_DIR"/moe-thinkoff.trace \
      --trace "$TRACE_DIR"/moe-efflow.trace \
      --ranks 2 --entry-rank 0 --expert-count 512 \
      --strategy coactivation --balance-tolerance 0 \
      --out "$EP_MAP" 2>&1
    printf '# map_sha256\t%s\n' "$(sha256sum "$EP_MAP" 2>/dev/null | cut -d' ' -f1)"
    # Also mint the frequency arm: the placement A/B wants a second measured strategy, not
    # just measured-vs-even, or "coactivation won" is untested against any other ordering.
    python3 "$SRC/tools/build_expert_placement_map.py" \
      --trace "$TRACE_DIR"/moe-thinkon.trace \
      --ranks 2 --entry-rank 0 --expert-count 512 \
      --strategy frequency --balance-tolerance 0 \
      --out "$OUT/ep-map-frequency.json" 2>&1
  } >> "$log_f" 2>&1
  log "cell $label rc=$?"
}

# =======================================================================================
# CELL E -- the 262,144 rung. BANKED AS BLOCKED, DELIBERATELY, with the reason stated rather
# than a script that cannot work.
#
# TP2 CANNOT REACH 262k ON THIS PROGRAM AND THE FAILURE IS MEASURED, NOT PREDICTED. Card 1 is
# FLAT at 43,603 MiB from fill=16,384 onward while card 0 absorbs every growing byte (the
# indexer block lists, the QSA raw-key host cache and its pooled_dev/raw_dev mirrors and
# idx_audit are structurally card-0 only), and TP2 post-load already costs card 0 +2,784 MiB
# over single-card (92,755 vs 89,971). The round-2 TP2 ladder OOM'd during the fill at
# 65,536 -> 81,920 with card 1 holding ~54 GiB that long context cannot use, while ONE card
# reaches ~731k on the ship-default cache. For depth, this program is a REGRESSION.
#
# So the 262k rung belongs to the single-card ladder, where it is already owed
# (PROFILE-10 section 6: proven to ALLOCATE, never filled and timed), and a TP2 262k cell
# would burn a ~23 minute prefill to reproduce a known OOM. It is not scripted here.
#
# Reopen condition, and nothing weaker: a program where the growing caches can live on card 1
# (i.e. a real residency partition AND a card-1-side indexer/raw-key path), at which point the
# cell is `--ladder 262144 --ladder-tp2` with a PAIR guard and the ascending-ladder trap in
# mind - `--ladder` allocates ONE state at the DEEPEST rung's capacity before the first rung
# runs, so a list whose top rung does not fit banks NOTHING (PROFILE-10 section 5). Run single
# rungs, never an ascending list, on a program near its ceiling.
# =======================================================================================
cell_262k(){
  log "cell E-262k BLOCKED BY DESIGN: TP2 OOMs below 100k (card 1 flat at 43,603 MiB, card 0 \
carries all growth); see EP2-DESIGN.md section 3. Not scripted."
  return 3
}

# =======================================================================================
usage(){
  cat <<'USAGE'
usage: q4e-ep2-cell.sh <cell>

  A            spec section profile, K=5, SHALLOW  -- the one cell worth running (single card).
               There is deliberately NO deep twin: --spec-profile is shallow-only and no
               --ladder-spec-profile exists. See the cell A header.
  B            two-card exactness gate, EVEN split (control)
  B-mapped     two-card exactness gate, measured map (needs D)
  C            t=1 plain decode A/B x3 reps, rotated arm order (needs D)
  D            collect router traces under --tp2 and mint the placement maps
  E            262k rung -- BLOCKED BY DESIGN, prints why

Read EP2-DESIGN.md first: two-card EP2 cannot reach the 200 tok/s target (needs 63.8% of the
round to be routed-expert work; it is 22.9-48.5%), so B/C/D are only worth a turn if two cards
are already committed for another reason. A is worth a turn on its own.
USAGE
}

case ${1:-} in
  A)        cell_spec_profile ;;
  B)        cell_exactness even ;;
  B-mapped) cell_exactness mapped ;;
  C)        for r in 0 1 2; do cell_decode_ab "$r"; done ;;
  D)        cell_mint_map ;;
  E)        cell_262k ;;
  *)        usage; exit 2 ;;
esac
