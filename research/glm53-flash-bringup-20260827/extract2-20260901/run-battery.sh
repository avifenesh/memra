#!/usr/bin/env bash
# lane/glm5-extract2 standing-battery runner (rig 5090, exactness only — rig law).
# Phase 2 of the extraction program. Same shape as phase 1's runner plus the two suites
# that landed since (glm5_ep_diet_doors_gpu, glm5_dedup_sched_gpu) and the ALIAS-COVERAGE
# arms: every renamed door is driven through BOTH names, and a disagreeing pair is proved
# to fall closed. The lane is renames + seam hoists with NO behavior change: every standing
# suite must stay green untouched, every engagement counter must keep counting, and every
# refusal must keep its bytes (the banked scripts set the ALIAS, so that is the
# receipt-comparable path; the general name is proved by its own arms).
#
# Runner law (moe-loc kda_fixture_gpu lesson): --include-ignored, NEVER bare --ignored,
# and a suite that reports "0 passed" FAILS the battery even on exit 0.
set -u
cd "$(dirname "$0")/../../.."
# SCCACHE-FLOCK TRAP (found live 2026-08-31, ~80 min rig deadlock across three queued
# lanes): a cargo build running UNDER the rig flock spawns the sccache daemon, which
# inherits the lock fd; when the flock'd command exits, the open file description — and
# the exclusive lock — live on in the daemon, and /proc/locks shows a DEAD pid as holder.
# Pre-warming the server OUTSIDE the lock means later builds spawn nothing under it.
sccache --start-server >/dev/null 2>&1 || true
OUT=research/glm53-flash-bringup-20260827/extract2-20260901/receipts
mkdir -p "$OUT" "$OUT/ppn-gate" "$OUT/hppn-gate" "$OUT/hbatch-gate"
fails=0

SUITES_STANDING="glm5_matvec_doors_gpu glm5_moe_loc_doors_gpu glm5_ep_diet_doors_gpu glm5_dedup_sched_gpu glm5_verify_batch_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu glm5_moe_epilogue_gpu glm5_mtp_head_gpu glm5_kpool_indexer_gpu hyper_connections_gpu hc_fused_pre_gpu hc_decode_ws_gpu mla_decode_split_gpu kda_fixture_gpu kda_fused_proj_gpu kda_fused_proj_bf16_gpu kda_quant_operand_gpu mla_gpu_forward"
SUITES_EXTRA="glm5_bf16_tc_trunk_prefill_gpu glm5_chunked_prime_gpu glm5_moe_grouped_prefill_gpu glm5_moe_residency_gpu glm5_routed_router_gpu mla_tc_prefill_gpu swiglu_preclamp_gpu glm5_admission_cost glm5_prime_capacity"
VISION_SHARD="$HOME/ai-ml/models/glm53-vision/model-00062-of-00062.safetensors"

run_suite() { # run_suite <log> <extra-env...> -- <suite>
  local log="$1"; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  local suite="$1"
  echo "########## $log :: $suite ${envs[*]:-} ##########"
  flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 "${envs[@]}" \
    timeout 5400 nice -n 5 cargo test -p memra-engine --test "$suite" -- --include-ignored --test-threads=1 \
    >"$OUT/$log" 2>&1
  local rc=$?
  echo "exit=$rc" >>"$OUT/$log"
  local result
  result=$(grep -E 'test result' "$OUT/$log" | tail -1)
  if [ "$rc" -ne 0 ]; then fails=$((fails+1)); echo "FAIL: $log ($result)"; return; fi
  # zero-test guard: a green run that ran NOTHING is a vacuous green, counted as failure
  if echo "$result" | grep -qE 'ok\. 0 passed'; then
    fails=$((fails+1)); echo "VACUOUS (0 tests ran): $log"; return
  fi
  echo "$result"
}

run_bin() { # run_bin <outdir> <log> <bin> <env-or-"-"> <args...>
  local dir="$1" log="$2" bin="$3" envs="$4"; shift 4
  echo "########## $dir/$log :: ${envs} $* ##########"
  local envlist=""
  [ "$envs" != "-" ] && envlist="$envs"
  # shellcheck disable=SC2086
  flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 $envlist \
    timeout 5400 nice -n 5 cargo run -q -p memra-engine --bin "$bin" -- "$@" \
    >"$OUT/$dir/$log" 2>&1
  local rc=$?
  echo "exit=$rc" >>"$OUT/$dir/$log"
  if [ "$rc" -ne 0 ]; then fails=$((fails+1)); echo "FAIL: $dir/$log"; else
    grep -cE 'PASS' "$OUT/$dir/$log" | sed 's/^/PASS lines: /'; fi
}

echo "== phase 1: standing + extra GPU suites (default arm — the ship program) =="
for s in $SUITES_STANDING $SUITES_EXTRA; do run_suite "$s.log" -- "$s"; done
if [ -f "$VISION_SHARD" ]; then
  run_suite glm5_vision_gpu.log MEMRA_GLM5_VISION_SHARD="$VISION_SHARD" -- glm5_vision_gpu
else
  echo "SKIP glm5_vision_gpu: shard artifact absent at $VISION_SHARD" | tee "$OUT/glm5_vision_gpu.log"
  fails=$((fails+1))
fi

# ALIAS COVERAGE lives INSIDE the arms, not in a process-wide env pin, and that is a
# design constraint rather than a shortcut: these doors are read PER CALL, so pinning a
# general name to `0` for a whole suite would DISAGREE with the suite's own alias-ON arm and
# fall the door closed — the pin would break the arm it was meant to cover. So:
#   * door H: glm5_moe_loc_doors_gpu carries three arms in one process — alias ON (the
#     banked path), GENERAL name ON, and a disagreeing pair asserted to leave the counter
#     FLAT while the value still lands through the shipped form.
#   * EP doors: glm5-tp-gate arms B2 (alias ON), B2G (general name ON, alias unset) and
#     B2X (disagreeing pair, dispatch counter 0).

echo "== phase 2: glm5-tp-gate P=16 N=12 (release — the ep-map arms exercise the renamed seam via the ALIAS name) =="
flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 \
  timeout 5400 nice -n 5 cargo run -q --release -p memra-engine --bin glm5-tp-gate -- 16 12 \
  >"$OUT/tp-gate-p16-n12.log" 2>&1
rc=$?
echo "exit=$rc" >>"$OUT/tp-gate-p16-n12.log"
if [ "$rc" -ne 0 ]; then fails=$((fails+1)); echo "FAIL: tp-gate"; else
  grep -E 'ALL ARMS|PASS' "$OUT/tp-gate-p16-n12.log" | tail -3; fi

echo "== phase 3: ppn / hppn / hbatch matrices (the banked arm shapes) =="
run_bin ppn-gate 10-n2-even.log      glm5-spec-ppn-gate -                     2 24 20
run_bin ppn-gate 11-n2-split1.log    glm5-spec-ppn-gate MEMRA_PP_SPLITS=1     2 24 20
run_bin ppn-gate 12-n2-split3.log    glm5-spec-ppn-gate MEMRA_PP_SPLITS=3     2 24 20
run_bin ppn-gate 13-n2-streams0.log  glm5-spec-ppn-gate MEMRA_PP_STREAMS=0    2 24 20
run_bin ppn-gate 14-n2-overlap0.log  glm5-spec-ppn-gate MEMRA_PP_OVERLAP=0    2 24 20
run_bin ppn-gate 16-n3-even.log      glm5-spec-ppn-gate -                     3 24 20
run_bin ppn-gate 17-n3-asym.log      glm5-spec-ppn-gate MEMRA_PP_SPLITS=1,3   3 24 20
run_bin ppn-gate 18-n3-streams0.log  glm5-spec-ppn-gate MEMRA_PP_STREAMS=0    3 24 20
# spec-trace twin: the ONE behavior-adjacent surface of this lane (the moved timers) —
# level 2 through the GENERAL flag name on the n2 arm; trace lines are shares, never perf.
run_bin ppn-gate 19-n2-spectrace2.log glm5-spec-ppn-gate MEMRA_SPEC_TRACE=2   2 24 20

run_bin hppn-gate 10-n2-even.log     glm5-hyper-ppn-gate -                    2 6 8
run_bin hppn-gate 11-n2-split1.log   glm5-hyper-ppn-gate MEMRA_PP_SPLITS=1    2 6 8
run_bin hppn-gate 12-n2-split3.log   glm5-hyper-ppn-gate MEMRA_PP_SPLITS=3    2 6 8
run_bin hppn-gate 13-n2-streams0.log glm5-hyper-ppn-gate MEMRA_PP_STREAMS=0   2 6 8
run_bin hppn-gate 14-n2-overlap0.log glm5-hyper-ppn-gate MEMRA_PP_OVERLAP=0   2 6 8
run_bin hppn-gate 15-n2-shard0.log   glm5-hyper-ppn-gate MEMRA_PP_SHARD=0     2 6 8
run_bin hppn-gate 16-n3-asym.log     glm5-hyper-ppn-gate MEMRA_PP_SPLITS=1,3  3 6 8
run_bin hppn-gate 17-n4-even.log     glm5-hyper-ppn-gate -                    4 6 8
run_bin hppn-gate 18-n4-streams0.log glm5-hyper-ppn-gate MEMRA_PP_STREAMS=0   4 6 8
run_bin hppn-gate 19-n2-longer.log   glm5-hyper-ppn-gate -                    2 16 24

run_bin hbatch-gate 10-b3-default.log       glm5-hyper-batch-gate -                  3 5 8 1
run_bin hbatch-gate 11-b8-wide.log          glm5-hyper-batch-gate -                  8 5 8 1
run_bin hbatch-gate 12-b2-longer.log        glm5-hyper-batch-gate -                  2 12 24 1
run_bin hbatch-gate 13-b3-ppn2.log          glm5-hyper-batch-gate -                  3 5 8 2
run_bin hbatch-gate 14-b3-ppn2-streams0.log glm5-hyper-batch-gate MEMRA_PP_STREAMS=0 3 5 8 2
run_bin hbatch-gate 15-b3-ppn4.log          glm5-hyper-batch-gate -                  3 5 8 4
run_bin hbatch-gate 16-b8-ppn2.log          glm5-hyper-batch-gate -                  8 5 8 2
run_bin hbatch-gate 17-b12.log              glm5-hyper-batch-gate -                  12 5 8 1
run_bin hbatch-gate 18-b15-cap.log          glm5-hyper-batch-gate -                  15 5 8 1
run_bin hbatch-gate 19-b15-ppn2.log         glm5-hyper-batch-gate -                  15 5 8 2

echo "== phase 4: memra-server suite (release, the ep-place shape) =="
flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 \
  timeout 5400 nice -n 5 cargo test -q --release -p memra-server \
  >"$OUT/memra-server-suite.log" 2>&1
rc=$?
echo "exit=$rc" >>"$OUT/memra-server-suite.log"
if [ "$rc" -ne 0 ]; then fails=$((fails+1)); echo "FAIL: memra-server suite"; else
  grep -E 'test result' "$OUT/memra-server-suite.log" | tail -2; fi

echo "== phase 5: engine lib unit tests + check-flags + fmt =="
flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 \
  nice -n 5 cargo test -q -p memra-engine --lib >"$OUT/engine-lib-units.log" 2>&1
rc=$?
echo "exit=$rc" >>"$OUT/engine-lib-units.log"
if [ "$rc" -ne 0 ]; then fails=$((fails+1)); echo "FAIL: engine lib units"; else
  grep -E 'test result' "$OUT/engine-lib-units.log" | tail -1; fi
bash tools/check-flags.sh >"$OUT/check-flags.log" 2>&1 || { fails=$((fails+1)); echo "FAIL: check-flags"; }
tail -2 "$OUT/check-flags.log"
cargo fmt -p memra-engine --check >"$OUT/fmt-check.log" 2>&1 || { fails=$((fails+1)); echo "FAIL: fmt"; }

echo "=========================================================="
if [ "$fails" -eq 0 ]; then echo "extract2 battery: ALL GATES PASS"; else echo "extract2 battery: $fails GATE(S) FAILED"; fi
exit "$fails"
