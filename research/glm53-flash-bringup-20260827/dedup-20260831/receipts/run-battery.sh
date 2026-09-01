#!/usr/bin/env bash
# lane/glm5-dedup standing-battery runner (rig 5090, exactness + counters only — rig law).
#
# STATE: RUN 2026-08-31 by the end-of-day debt lane after the owner released the rig. Written
# UNRUN + deferred per the owner rig-hold order; the deferral is now paid. The two doors stay
# DEFAULT OFF regardless of this exit code (the identity is structural, the box prices the flip) —
# green here only clears the bringup merge. See ../LANE.md §6 for the receipt table.
#
# Amended on the debt run: `run()` now asserts a NON-ZERO passed count, not just exit 0. The
# original body treated rc=0 as PASS, which is exactly the "ok. 0 passed; 3 filtered out" shape
# the header below warns about — the assertion has to be in the code, not only in the comment.
#
# Phase 1: every standing GPU suite on the DEFAULT arm — this lane's two doors pinned =0, which is
#          the SHIP arm (doors T/X/K/W are default ON at this base, so "no flags" is not "no
#          doors"; door M pinned =0 too, never unset — the moe-loc §4.5 lesson).
# Phase 2: the walk suites with doors E + E-down ON. This is where the transposed grids actually
#          run inside a verify walk, and it is the arm the identity gates CANNOT substitute for:
#          gate 2/3 prove the kernels, phase 2 proves the walk they sit in is unperturbed.
# Phase 3: the walk suites with doors E + E-down ON *composed with door D* (device-built tables AND
#          a device-built order plane) — the serving shape the box will price, since D is a live
#          1.0154x win. Door H rides along for the same reason.
# Phase 4: the down half ALONE, so a split verdict on the two halves stays expressible if phase 2
#          or 3 ever regresses (the two doors carry different DRAM-stream risk, LANE.md §4.3).
set -u
cd "$(dirname "$0")/../../../.."
OUT=research/glm53-flash-bringup-20260827/dedup-20260831/receipts
fails=0
SUITES_ALL="glm5_dedup_sched_gpu glm5_moe_loc_doors_gpu glm5_verify_batch_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu glm5_moe_epilogue_gpu glm5_mtp_head_gpu glm5_kpool_indexer_gpu glm5_matvec_doors_gpu hyper_connections_gpu hc_fused_pre_gpu hc_decode_ws_gpu mla_decode_split_gpu kda_fixture_gpu kda_fused_proj_gpu kda_fused_proj_bf16_gpu kda_quant_operand_gpu mla_gpu_forward"
SUITES_WALK="glm5_verify_batch_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu glm5_moe_epilogue_gpu"

# --include-ignored, NOT --ignored: `--ignored` filters out every non-ignored test in a suite, which
# is how moe-loc found kda_fixture_gpu reporting "ok. 0 passed; 3 filtered out" — a suite that ran
# NOTHING, banked as green.
run() { # run <log> <extra-env...> -- <suite>
  local log="$1"; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  local suite="$1"
  echo "########## $log :: $suite ${envs[*]:-} ##########"
  flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 "${envs[@]}" \
    timeout 3600 nice -n 5 cargo test -p memra-engine --test "$suite" -- --include-ignored --test-threads=1 \
    >"$OUT/$log" 2>&1
  local rc=$?
  echo "exit=$rc" >>"$OUT/$log"
  # CAPTURE-THEN-GATE, and assert a NON-ZERO passed count: exit 0 with "0 passed; N filtered
  # out" is the rotted-gate shape moe-loc caught on kda_fixture_gpu. Sum every `test result`
  # line's passed count (a suite can report more than one); zero passed is a FAIL even at rc=0.
  local line passed
  line="$(grep -E '^test result' "$OUT/$log" | tail -1)"
  passed="$(grep -Eo '[0-9]+ passed' "$OUT/$log" | awk '{s+=$1} END {print s+0}')"
  echo "passed_total=$passed" >>"$OUT/$log"
  if [ "$rc" -ne 0 ]; then
    fails=$((fails+1)); echo "FAIL: $log (exit=$rc) ${line:-<no test result line>}"
  elif [ "$passed" -eq 0 ]; then
    fails=$((fails+1)); echo "FAIL: $log ZERO TESTS RAN (exit=0, passed=0) ${line:-<no test result line>}"
  else
    echo "PASS: $log passed=$passed | $line"
  fi
}

OFF="MEMRA_MOE_VROWS_DEDUP_ORDER=0 MEMRA_MOE_VROWS_DOWN_TMAJ=0 MEMRA_MOE_VROWS_PACK=0"
for s in $SUITES_ALL; do run "$s.log" $OFF -- "$s"; done
for s in $SUITES_WALK; do
  run "compose-$s.log" MEMRA_MOE_VROWS_DEDUP_ORDER=1 MEMRA_MOE_VROWS_DOWN_TMAJ=1 \
    MEMRA_MOE_VROWS_PACK=0 MEMRA_MOE_VROWS_DEV_TABLES=0 -- "$s"
done
for s in $SUITES_WALK; do
  run "composeD-$s.log" MEMRA_MOE_VROWS_DEDUP_ORDER=1 MEMRA_MOE_VROWS_DOWN_TMAJ=1 \
    MEMRA_MOE_VROWS_PACK=0 MEMRA_MOE_VROWS_DEV_TABLES=1 MEMRA_GLM5_HTOD_DIET=1 -- "$s"
done
for s in $SUITES_WALK; do
  run "downonly-$s.log" MEMRA_MOE_VROWS_DEDUP_ORDER=0 MEMRA_MOE_VROWS_DOWN_TMAJ=1 \
    MEMRA_MOE_VROWS_PACK=0 -- "$s"
done
echo "=========================================================="
if [ "$fails" -eq 0 ]; then echo "dedup battery: ALL SUITES PASS"; else echo "dedup battery: $fails SUITE(S) FAILED"; fi
exit "$fails"
