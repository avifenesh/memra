#!/usr/bin/env bash
# lane/glm5-ep-diet standing-battery runner (rig 5090, exactness + counters only — rig law).
#
# Phase 1: every standing GPU suite on the SHIP arm with this lane's two doors PINNED =0
#          (the matvec doors T/X/K/W are default ON at this base, so this is the shipped
#          program — the correct control), plus the new glm5_ep_diet_doors_gpu gate (which
#          drives its own kernels directly; the flags gate nothing inside it).
#
# There is NO standing-suite compose arm for these doors BY DESIGN: an enabled
# MEMRA_GLM5_EP_DIET / MEMRA_GLM5_EP_GROUPED_PRIME without MEMRA_GLM5_TP refuses at load
# on glm5-class plans (the silent-even-split trap, extended). The doors-ON walks are
# glm5-tp-gate arms B2/B3/M2/R2D/R3D, run by run-tp-gate.sh from the tp2 lane dir.
set -u
cd "$(dirname "$0")/../../../.."
OUT=research/glm53-flash-bringup-20260827/ep-diet-20260831/receipts
fails=0
SUITES_ALL="glm5_ep_diet_doors_gpu glm5_moe_loc_doors_gpu glm5_verify_batch_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu glm5_moe_epilogue_gpu glm5_mtp_head_gpu glm5_kpool_indexer_gpu glm5_matvec_doors_gpu hyper_connections_gpu hc_fused_pre_gpu hc_decode_ws_gpu mla_decode_split_gpu kda_fixture_gpu kda_fused_proj_gpu kda_fused_proj_bf16_gpu kda_quant_operand_gpu mla_gpu_forward"
run() { # run <log> <extra-env...> -- <suite>
  local log="$1"; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  local suite="$1"
  echo "########## $log :: $suite ${envs[*]:-} ##########"
  flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 \
    MEMRA_GLM5_EP_DIET=0 MEMRA_GLM5_EP_GROUPED_PRIME=0 "${envs[@]}" \
    timeout 3600 nice -n 5 cargo test -p memra-engine --test "$suite" -- --include-ignored --test-threads=1 \
    >"$OUT/$log" 2>&1
  local rc=$?
  echo "exit=$rc" >>"$OUT/$log"
  if [ "$rc" -ne 0 ]; then fails=$((fails+1)); echo "FAIL: $log"; else
    grep -E 'test result' "$OUT/$log" | tail -1; fi
}
for s in $SUITES_ALL; do run "$s.log" -- "$s"; done
echo "=========================================================="
if [ "$fails" -eq 0 ]; then echo "ep-diet battery: ALL SUITES PASS"; else echo "ep-diet battery: $fails SUITE(S) FAILED"; fi
exit "$fails"
