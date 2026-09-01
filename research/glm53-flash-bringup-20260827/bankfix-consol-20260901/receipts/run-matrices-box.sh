#!/usr/bin/env bash
# bank-defect consolidation: the split-gate PPN matrices on box B, all four cards.
#
# Arm shapes are copied VERBATIM from ../../dedup-20260831/receipts/run-matrices.sh (which copied
# them from moe-loc-20260831) so these receipts are comparable arm-for-arm with the two lanes this
# merge composes. What is new here is only the tree: bringup + main, one binary per gate.
#
# STAGE COUNTS ARE REAL HERE. On the rig every "n2/n3/n4" arm emulated N pipeline stages on one
# card; this box has four, so `CUDA_VISIBLE_DEVICES=0,1,2,3` lets the gates place stages on distinct
# devices. That is a STRICTLY WIDER shape than the banked rig arms, not a substitute for them: a
# cross-device boundary copy is a different program from a same-device one.
#
# WHAT ENGAGEMENT TO EXPECT, stated up front so a silent log is not read as a pass: the
# MEMRA_MOE_VROWS_* doors engage ONLY in the SPEC verify walk, so they are STRUCTURALLY SILENT in
# glm5-hyper-ppn-gate and glm5-hyper-batch-gate. Those compose arms are no-perturbation arms; the
# engagement receipt lives in the spec-ppn compose arms and in run-battery-box.sh phases 2-4.
#
# PASS-LINE COUNTS ARE ASSERTED, not merely printed (the moe-loc runner echoed a count that could
# have been 0 with exit 0): spec-ppn 23, hyper-ppn 6, hyper-batch 3.
# Whole-box lock: these arms own every card, so nothing from run-battery-box.sh may overlap.
set -u
cd "$(dirname "$0")/../../../.."
BASE="${BANKFIX_OUT:-/root/out-bankfix}/matrices"
mkdir -p "$BASE"
fails=0
STARTED=""

run() { # run <outdir> <log> <bin> <expected-pass-lines> <env-or-"-"> <args...>
    local dir="$1" log="$2" bin="$3" want="$4" envs="$5"; shift 5
    mkdir -p "$BASE/$dir"
    # TRUNCATE the summary on the first arm of a matrix (moe-loc: a `tee -a` across re-runs left a
    # stale FAIL line in a banked receipt, reading exactly like a live failure).
    case " $STARTED " in
        *" $dir "*) ;;
        *) : >"$BASE/$dir/matrix.out"; STARTED="$STARTED $dir";;
    esac
    echo "########## $log :: ${envs} $* ##########" | tee -a "$BASE/$dir/matrix.out"
    local envlist=""
    [ "$envs" != "-" ] && envlist="$envs"
    # CAPTURE-THEN-GATE: redirect to a file, take rc, then judge. No pipe on the failable step.
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=0,1,2,3 flock /tmp/memra-boxB-all.lock \
        env NVIDIA_TF32_OVERRIDE=0 $envlist \
        timeout 5400 nice -n 5 ./target/debug/"$bin" "$@" \
        >"$BASE/$dir/$log" 2>&1
    local rc=$?
    echo "exit=$rc" >>"$BASE/$dir/$log"
    local got
    got="$(grep -cE 'gate PASS' "$BASE/$dir/$log")"
    echo "pass_lines=$got want=$want" >>"$BASE/$dir/$log"
    if [ "$rc" -ne 0 ]; then
        fails=$((fails+1)); echo "FAIL: $dir/$log (exit=$rc, pass_lines=$got)" | tee -a "$BASE/$dir/matrix.out"
    elif [ "$got" -ne "$want" ]; then
        fails=$((fails+1))
        echo "FAIL: $dir/$log WRONG PASS-LINE COUNT (exit=0, pass_lines=$got, want=$want)" | tee -a "$BASE/$dir/matrix.out"
    else
        echo "PASS: $dir/$log pass_lines=$got" | tee -a "$BASE/$dir/matrix.out"
    fi
    tail -3 "$BASE/$dir/$log" >>"$BASE/$dir/matrix.out"
}

OFF="MEMRA_MOE_VROWS_DEDUP_ORDER=0 MEMRA_MOE_VROWS_DOWN_TMAJ=0 MEMRA_MOE_VROWS_PACK=0"
E="MEMRA_MOE_VROWS_DEDUP_ORDER=1 MEMRA_MOE_VROWS_DOWN_TMAJ=1 MEMRA_MOE_VROWS_PACK=0 MEMRA_MOE_VROWS_DEV_TABLES=0"
EDH="MEMRA_MOE_VROWS_DEDUP_ORDER=1 MEMRA_MOE_VROWS_DOWN_TMAJ=1 MEMRA_MOE_VROWS_PACK=0 MEMRA_MOE_VROWS_DEV_TABLES=1 MEMRA_GLM5_HTOD_DIET=1"

echo "=== bankfix PPN matrices on box B: $(date -u +%Y-%m-%dT%H:%M:%SZ) lane sha=$(git rev-parse HEAD) ==="

# ---- glm5-spec-ppn-gate (23 PASS lines/arm) ----
run ppn-gate 10-n2-even.log     glm5-spec-ppn-gate 23 "$OFF"                       2 24 20
run ppn-gate 11-n2-split1.log   glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_SPLITS=1"     2 24 20
run ppn-gate 12-n2-split3.log   glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_SPLITS=3"     2 24 20
run ppn-gate 13-n2-streams0.log glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_STREAMS=0"    2 24 20
run ppn-gate 14-n2-overlap0.log glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_OVERLAP=0"    2 24 20
run ppn-gate 16-n3-even.log     glm5-spec-ppn-gate 23 "$OFF"                       3 24 20
run ppn-gate 17-n3-asym.log     glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_SPLITS=1,3"   3 24 20
run ppn-gate 18-n3-streams0.log glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_STREAMS=0"    3 24 20
run ppn-gate compose-n2-even-doors-E.log   glm5-spec-ppn-gate 23 "$E"                     2 24 20
run ppn-gate compose-n3-even-doors-E.log   glm5-spec-ppn-gate 23 "$E"                     3 24 20
run ppn-gate compose-n3-asym-doors-E.log   glm5-spec-ppn-gate 23 "$E MEMRA_PP_SPLITS=1,3" 3 24 20
run ppn-gate compose-n2-even-doors-EDH.log glm5-spec-ppn-gate 23 "$EDH"                   2 24 20
run ppn-gate compose-n3-even-doors-EDH.log glm5-spec-ppn-gate 23 "$EDH"                   3 24 20

# ---- glm5-hyper-ppn-gate (6 PASS lines/arm) ----
run hppn-gate 10-n2-even.log     glm5-hyper-ppn-gate 6 "$OFF"                      2 6 8
run hppn-gate 11-n2-split1.log   glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_SPLITS=1"    2 6 8
run hppn-gate 12-n2-split3.log   glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_SPLITS=3"    2 6 8
run hppn-gate 13-n2-streams0.log glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_STREAMS=0"   2 6 8
run hppn-gate 14-n2-overlap0.log glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_OVERLAP=0"   2 6 8
run hppn-gate 15-n2-shard0.log   glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_SHARD=0"     2 6 8
run hppn-gate 16-n3-asym.log     glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_SPLITS=1,3"  3 6 8
run hppn-gate 17-n4-even.log     glm5-hyper-ppn-gate 6 "$OFF"                      4 6 8
run hppn-gate 18-n4-streams0.log glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_STREAMS=0"   4 6 8
run hppn-gate 19-n2-longer.log   glm5-hyper-ppn-gate 6 "$OFF"                      2 16 24
run hppn-gate compose-n2-even-doors-E.log   glm5-hyper-ppn-gate 6 "$E"             2 6 8
run hppn-gate compose-n2-even-doors-EDH.log glm5-hyper-ppn-gate 6 "$EDH"           2 6 8

# ---- glm5-hyper-batch-gate (3 PASS lines/arm), B P N ppn ----
run hbatch-gate 10-b3-default.log       glm5-hyper-batch-gate 3 "$OFF"                     3 5 8 1
run hbatch-gate 11-b8-wide.log          glm5-hyper-batch-gate 3 "$OFF"                     8 5 8 1
run hbatch-gate 12-b2-longer.log        glm5-hyper-batch-gate 3 "$OFF"                     2 12 24 1
run hbatch-gate 13-b3-ppn2.log          glm5-hyper-batch-gate 3 "$OFF"                     3 5 8 2
run hbatch-gate 14-b3-ppn2-streams0.log glm5-hyper-batch-gate 3 "$OFF MEMRA_PP_STREAMS=0"  3 5 8 2
run hbatch-gate 15-b3-ppn4.log          glm5-hyper-batch-gate 3 "$OFF"                     3 5 8 4
run hbatch-gate 16-b8-ppn2.log          glm5-hyper-batch-gate 3 "$OFF"                     8 5 8 2
run hbatch-gate 17-b12.log              glm5-hyper-batch-gate 3 "$OFF"                     12 5 8 1
run hbatch-gate 18-b15-cap.log          glm5-hyper-batch-gate 3 "$OFF"                     15 5 8 1
run hbatch-gate 19-b15-ppn2.log         glm5-hyper-batch-gate 3 "$OFF"                     15 5 8 2
run hbatch-gate compose-b3-doors-E.log   glm5-hyper-batch-gate 3 "$E"                      3 5 8 1
run hbatch-gate compose-b3-doors-EDH.log glm5-hyper-batch-gate 3 "$EDH"                    3 5 8 1

echo "=========================================================="
echo "cards after: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' | ')"
if [ "$fails" -eq 0 ]; then echo "bankfix matrices: ALL ARMS PASS"; else echo "bankfix matrices: $fails ARM(S) FAILED"; fi
exit "$fails"
