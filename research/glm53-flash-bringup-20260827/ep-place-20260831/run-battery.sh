#!/usr/bin/env bash
# lane/glm5-ep-place standing-battery runner (rig 5090, exactness only — rig law).
# The lane's default arm changes NOTHING in the shipped program (tap OFF = no writer;
# map absent = the even split, byte-unchanged; the EP local_of indirection is TP-only,
# a refused door in serving) — every standing suite must stay green untouched.
# The lane-specific arms (map skew, corrupt-map red, tap identity + row count, map
# fail-closed refusals) live inside glm5-tp-gate (arms M/H1-H4/T/R4), run separately.
set -u
cd "$(dirname "$0")/../../.."
OUT=research/glm53-flash-bringup-20260827/ep-place-20260831/gates
fails=0
SUITES_ALL="glm5_matvec_doors_gpu glm5_verify_batch_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu glm5_moe_epilogue_gpu glm5_mtp_head_gpu glm5_kpool_indexer_gpu hyper_connections_gpu hc_fused_pre_gpu hc_decode_ws_gpu mla_decode_split_gpu kda_fixture_gpu kda_fused_proj_gpu kda_fused_proj_bf16_gpu kda_quant_operand_gpu mla_gpu_forward"
run() { # run <log> -- <suite>
  local log="$1"; shift; shift
  local suite="$1"
  echo "########## $log :: $suite ##########"
  flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 \
    timeout 3600 nice -n 5 cargo test -p memra-engine --test "$suite" -- --ignored --test-threads=1 \
    >"$OUT/$log" 2>&1
  local rc=$?
  echo "exit=$rc" >>"$OUT/$log"
  if [ "$rc" -ne 0 ]; then fails=$((fails+1)); echo "FAIL: $log"; else
    grep -E 'test result' "$OUT/$log" | tail -1; fi
}
for s in $SUITES_ALL; do run "$s.log" -- "$s"; done
echo "=========================================================="
if [ "$fails" -eq 0 ]; then echo "ep-place battery: ALL SUITES PASS"; else echo "ep-place battery: $fails SUITE(S) FAILED"; fi
exit "$fails"
