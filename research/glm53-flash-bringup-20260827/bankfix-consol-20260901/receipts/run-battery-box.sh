#!/usr/bin/env bash
# bank-defect consolidation: the glm5 standing GPU battery on box B (4x RTX PRO 6000 Blackwell WS).
#
# WHY ON THE BOX AND NOT THE RIG: the merge carries BOTH feature sets (the QT_NVFP4_V2 in_f
# scale-fetch fix from main, and the bringup's dedup/matvec/moe-loc/ep-diet doors), and the defect
# it closes lived in the PREFILL grouped GEMM — a path the rig's 24 GB laptop card cannot hold at
# the real geometry. The rig also throttles (~52% clock), so it is exactness-only by law; nothing
# here is timed, but the shapes are real.
#
# STRUCTURE
#   Phase 1  every standing suite on the SHIP arm, doors PINNED =0 (never merely unset — the
#            moe-loc §4.5 lesson: doors T/X/K passed vacuously because their reference arms were
#            unset-shaped, so "no flags" is not "no doors" at this base).
#   Phase 2  the walk suites with the dedup doors E + E-down ON: the transposed schedules actually
#            executing inside a verify walk, which the identity gates cannot substitute for.
#   Phase 3  phase 2 composed with door D (device-built tables) + door H (HtoD diet) — the serving
#            shape.
#   Phase 4  the down half ALONE, so a split verdict stays expressible.
#   Phase 5  glm5-tp-gate, every arm in one process (A-H5 + the four RED arms).
#
# ASSERTIONS, in the code and not only in the comments:
#   * --include-ignored, NEVER --ignored (`--ignored` filters out every non-ignored test, which is
#     how moe-loc found kda_fixture_gpu reporting "ok. 0 passed; 3 filtered out" — a suite that ran
#     NOTHING, banked as green).
#   * a NON-ZERO passed count is required; exit 0 with passed=0 is a FAIL.
#   * capture-then-gate: the failable step writes a file and rc is taken before anything judges it.
#     No pipe on the failable step.
#
# PARALLELISM: every suite here is single-GPU (the TP suites emulate two ranks with two contexts on
# one card), so the four cards run four independent serial streams, each pinned by
# CUDA_VISIBLE_DEVICES and each holding its OWN card lock. The multi-card matrices are a separate
# script and must not overlap this one.
set -u
cd "$(dirname "$0")/../../../.."
OUT="${BANKFIX_OUT:-/root/out-bankfix}/battery"
mkdir -p "$OUT"
LANE_SHA="$(git rev-parse HEAD)"

# Doors pinned =0: the SHIP arm for the dedup lane's two doors plus door M (matvec), never unset.
OFF="MEMRA_MOE_VROWS_DEDUP_ORDER=0 MEMRA_MOE_VROWS_DOWN_TMAJ=0 MEMRA_MOE_VROWS_PACK=0 MEMRA_MOE_VROWS_DEV_TABLES=0"
E="MEMRA_MOE_VROWS_DEDUP_ORDER=1 MEMRA_MOE_VROWS_DOWN_TMAJ=1 MEMRA_MOE_VROWS_PACK=0 MEMRA_MOE_VROWS_DEV_TABLES=0"
EDH="MEMRA_MOE_VROWS_DEDUP_ORDER=1 MEMRA_MOE_VROWS_DOWN_TMAJ=1 MEMRA_MOE_VROWS_PACK=0 MEMRA_MOE_VROWS_DEV_TABLES=1 MEMRA_GLM5_HTOD_DIET=1"
DOWN="MEMRA_MOE_VROWS_DEDUP_ORDER=0 MEMRA_MOE_VROWS_DOWN_TMAJ=1 MEMRA_MOE_VROWS_PACK=0 MEMRA_MOE_VROWS_DEV_TABLES=0"

# The bank-defect's own neighbourhood first (grouped prefill, the routed FFN, the doors that read
# the v2 bank), then the rest of the standing set.
SUITES_ALL="glm5_moe_grouped_prefill_gpu glm5_ep_diet_doors_gpu glm5_dedup_sched_gpu \
glm5_moe_loc_doors_gpu glm5_matvec_doors_gpu glm5_verify_batch_gpu glm5_tparallel_verify_gpu \
glm5_spec_session_gpu glm5_dflash_session_gpu glm5_moe_epilogue_gpu glm5_mtp_head_gpu \
glm5_kpool_indexer_gpu glm5_routed_router_gpu glm5_moe_residency_gpu glm5_chunked_prime_gpu \
mla_tc_prefill_gpu mla_decode_split_gpu mla_gpu_forward hyper_connections_gpu hc_fused_pre_gpu \
hc_decode_ws_gpu kda_fixture_gpu kda_fused_proj_gpu kda_fused_proj_bf16_gpu kda_quant_operand_gpu \
swiglu_preclamp_gpu glm5_bf16_tc_trunk_prefill_gpu"
SUITES_WALK="glm5_verify_batch_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu \
glm5_dflash_session_gpu glm5_moe_epilogue_gpu glm5_dedup_sched_gpu"

# run <card> <log> <env...> -- <suite>
run() {
    local card="$1" log="$2"; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    local suite="$1"
    local f="$OUT/$log"
    echo "########## $log :: $suite (card $card) ${envs[*]:-} ##########" >"$f"
    CUDA_VISIBLE_DEVICES="$card" flock "/tmp/memra-boxB-card$card.lock" \
        env NVIDIA_TF32_OVERRIDE=0 "${envs[@]}" \
        timeout 5400 nice -n 5 cargo test -p memra-engine --test "$suite" -- \
        --include-ignored --test-threads=1 >>"$f" 2>&1
    local rc=$?
    echo "exit=$rc" >>"$f"
    local line passed
    line="$(grep -E '^test result' "$f" | tail -1)"
    passed="$(grep -Eo '[0-9]+ passed' "$f" | awk '{s+=$1} END {print s+0}')"
    echo "passed_total=$passed" >>"$f"
    if [ "$rc" -ne 0 ]; then
        echo "FAIL: $log (exit=$rc) ${line:-<no test result line>}"
    elif [ "$passed" -eq 0 ]; then
        echo "FAIL: $log ZERO TESTS RAN (exit=0, passed=0) ${line:-<no test result line>}"
    else
        echo "PASS: $log passed=$passed | $line"
    fi
}

# One card's whole queue, serial, in a subshell; the verdict lines go to a per-card summary.
stream() { # stream <card> <suite...>
    local card="$1"; shift
    for s in "$@"; do
        run "$card" "$s.log" $OFF -- "$s"
    done
}

echo "=== bankfix glm5 GPU battery on box B: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "lane sha=$LANE_SHA"
echo "cards: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tr '\n' ' | ')"

# ---- phases 1: fan the standing suites across the four cards ----
i=0
c0=(); c1=(); c2=(); c3=()
for s in $SUITES_ALL; do
    case $((i % 4)) in
        0) c0+=("$s");; 1) c1+=("$s");; 2) c2+=("$s");; 3) c3+=("$s");;
    esac
    i=$((i+1))
done
stream 0 "${c0[@]}" >"$OUT/summary-card0.txt" 2>&1 &
p0=$!
stream 1 "${c1[@]}" >"$OUT/summary-card1.txt" 2>&1 &
p1=$!
stream 2 "${c2[@]}" >"$OUT/summary-card2.txt" 2>&1 &
p2=$!
stream 3 "${c3[@]}" >"$OUT/summary-card3.txt" 2>&1 &
p3=$!
wait $p0 $p1 $p2 $p3

# ---- phases 2-4: the compose arms, walk suites only ----
compose_stream() { # compose_stream <card> <prefix> <envstring>
    local card="$1" prefix="$2" envs="$3"
    for s in $SUITES_WALK; do
        # shellcheck disable=SC2086
        run "$card" "$prefix-$s.log" $envs -- "$s"
    done
}
compose_stream 0 compose   "$E"    >"$OUT/summary-compose.txt" 2>&1 &
q0=$!
compose_stream 1 composeD  "$EDH"  >"$OUT/summary-composeD.txt" 2>&1 &
q1=$!
compose_stream 2 downonly  "$DOWN" >"$OUT/summary-downonly.txt" 2>&1 &
q2=$!
wait $q0 $q1 $q2

# ---- phase 5: glm5-tp-gate, all arms in one process (card 3, its own lock) ----
tpg="$OUT/tp-gate.log"
echo "########## glm5-tp-gate 16 12 (all arms, card 3) ##########" >"$tpg"
CUDA_VISIBLE_DEVICES=3 flock /tmp/memra-boxB-card3.lock \
    env NVIDIA_TF32_OVERRIDE=0 timeout 5400 nice -n 5 ./target/debug/glm5-tp-gate 16 12 \
    >>"$tpg" 2>&1
tp_rc=$?
echo "exit=$tp_rc" >>"$tpg"
if [ "$tp_rc" -eq 0 ] && grep -q "ALL ARMS PASS" "$tpg"; then
    echo "PASS: tp-gate $(grep -o 'ALL ARMS PASS.*' "$tpg" | head -1)" >"$OUT/summary-tp-gate.txt"
else
    echo "FAIL: tp-gate (exit=$tp_rc) $(tail -2 "$tpg" | tr '\n' ' ')" >"$OUT/summary-tp-gate.txt"
fi

# ---- verdict ----
cat "$OUT"/summary-*.txt >"$OUT/ALL.txt"
fails="$(grep -c '^FAIL' "$OUT/ALL.txt")"
passes="$(grep -c '^PASS' "$OUT/ALL.txt")"
echo "=========================================================="
grep '^FAIL' "$OUT/ALL.txt" || true
echo "cards after: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' | ')"
if [ "$fails" -eq 0 ]; then
    echo "bankfix GPU battery: ALL $passes ARMS PASS"
else
    echo "bankfix GPU battery: $fails ARM(S) FAILED ($passes passed)"
fi
exit "$fails"
