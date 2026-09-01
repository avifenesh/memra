#!/usr/bin/env bash
# lane/glm5-door-r standing-battery runner (rig 5090, exactness + counters only — rig law).
#
# Phase 1: every standing GPU suite on the DEFAULT arm (door R OFF — the shipped program must
#          be byte-for-byte untouched by this lane), incl. glm5_matvec_doors_gpu (which now
#          carries the door-R gate and drives its own flags per call) and the moe-loc gate.
# Phase 2: the walk suites with MEMRA_BF16_TCOLS_RED_FUSED=1 (the compose arm) — the _rf
#          twins run inside the real verify walk and drafter head; the boot announce line in
#          these logs is the walk-level engagement receipt.
#
# NOTE the matvec doors T/X/K/W are DEFAULT ON at this base, so phase 1 is the SHIP arm — the
# correct control. Door M stays OFF (=0 pinned where an arm needs it, never unset). Run with
# --include-ignored, never --ignored (moe-loc §4.2: --ignored ran ZERO tests in five suites).
set -u
cd "$(dirname "$0")/../../../.."
OUT=research/glm53-flash-bringup-20260827/door-r-20260831/receipts
fails=0
SUITES_ALL="glm5_matvec_doors_gpu glm5_moe_loc_doors_gpu glm5_verify_batch_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu glm5_moe_epilogue_gpu glm5_mtp_head_gpu glm5_kpool_indexer_gpu hyper_connections_gpu hc_fused_pre_gpu hc_decode_ws_gpu mla_decode_split_gpu kda_fixture_gpu kda_fused_proj_gpu kda_fused_proj_bf16_gpu kda_quant_operand_gpu mla_gpu_forward"
SUITES_WALK="glm5_verify_batch_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu glm5_moe_epilogue_gpu"
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
  if [ "$rc" -ne 0 ]; then fails=$((fails+1)); echo "FAIL: $log"; else
    grep -E 'test result' "$OUT/$log" | tail -1; fi
}
for s in $SUITES_ALL; do run "$s.log" -- "$s"; done
for s in $SUITES_WALK; do
  run "compose-$s.log" MEMRA_BF16_TCOLS_RED_FUSED=1 MEMRA_MOE_VROWS_PACK=0 -- "$s"
done
echo "=========================================================="
if [ "$fails" -eq 0 ]; then echo "door-r battery: ALL SUITES PASS"; else echo "door-r battery: $fails SUITE(S) FAILED"; fi
exit "$fails"
