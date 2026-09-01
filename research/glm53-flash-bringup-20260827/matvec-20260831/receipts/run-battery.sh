#!/usr/bin/env bash
# lane/glm5-matvec standing-battery runner (rig 5090, exactness only — rig law).
# Phase 1: every standing GPU suite on the DEFAULT arm (all five doors OFF — the shipped
# program must be byte-for-byte untouched), plus the new doors gate (which drives its own
# flags per call). Phase 2: the walk suites again with ALL FIVE DOORS ON (compose arm).
# Engagement scope, stated: on the mini fixtures doors W and M engage in the walk suites
# (pool hits + NVFP4 vrows pair); doors T/X need >=2M-element bf16 tensors and door K needs
# n_cols>=16384, so their exactness is carried by glm5_matvec_doors_gpu's own fixtures.
set -u
cd "$(dirname "$0")/../../../.."
OUT=research/glm53-flash-bringup-20260827/matvec-20260831/receipts
fails=0
SUITES_ALL="glm5_matvec_doors_gpu glm5_verify_batch_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu glm5_moe_epilogue_gpu glm5_mtp_head_gpu glm5_kpool_indexer_gpu hyper_connections_gpu hc_fused_pre_gpu hc_decode_ws_gpu mla_decode_split_gpu kda_fixture_gpu kda_fused_proj_gpu kda_fused_proj_bf16_gpu kda_quant_operand_gpu mla_gpu_forward"
SUITES_COMPOSE="glm5_verify_batch_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu glm5_moe_epilogue_gpu"
run() { # run <log> <extra-env...> -- <suite>
  local log="$1"; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  local suite="$1"
  echo "########## $log :: $suite ${envs[*]:-} ##########"
  flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 "${envs[@]}" \
    timeout 3600 nice -n 5 cargo test -p memra-engine --test "$suite" -- --ignored --test-threads=1 \
    >"$OUT/$log" 2>&1
  local rc=$?
  echo "exit=$rc" >>"$OUT/$log"
  if [ "$rc" -ne 0 ]; then fails=$((fails+1)); echo "FAIL: $log"; else
    grep -E 'test result' "$OUT/$log" | tail -1; fi
}
for s in $SUITES_ALL; do run "$s.log" -- "$s"; done
for s in $SUITES_COMPOSE; do
  run "compose-$s.log" MEMRA_BF16_TCOLS_WIDE=1 MEMRA_BF16_TCOLS_X1=1 MEMRA_MOE_VROWS_PACK=1 MEMRA_TOPK_SHARDS=1 MEMRA_GLM5_VERIFY_WS=1 -- "$s"
done
echo "=========================================================="
if [ "$fails" -eq 0 ]; then echo "matvec battery: ALL SUITES PASS"; else echo "matvec battery: $fails SUITE(S) FAILED"; fi
exit "$fails"
