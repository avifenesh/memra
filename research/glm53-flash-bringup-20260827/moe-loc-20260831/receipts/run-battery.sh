#!/usr/bin/env bash
# lane/glm5-moe-loc standing-battery runner (rig 5090, exactness + counters only — rig law).
#
# Phase 1: every standing GPU suite on the DEFAULT arm (doors D/H OFF and instrument S off — the
#          shipped program must be byte-for-byte untouched by this lane), plus the new
#          glm5_moe_loc_doors_gpu gate (which drives its own flags per call).
# Phase 2: the walk suites again with doors D + H ON (the compose arm) — this is where door D's
#          device routing arm and door H's two substitutions actually run inside a verify walk.
# Phase 3: the walk suites with the S instrument on (and door D pinned OFF, since S counts the
#          HOST selection door D removes) — proves the counter moves and the arm still matches.
#
# NOTE the matvec doors are DEFAULT ON at this base (T/X/K/W), so phase 1 is not a "no doors"
# arm — it is the SHIP arm, which is the correct control for this lane. Door M stays OFF (=0
# pinned where an arm needs it, never unset — the lane rule).
set -u
cd "$(dirname "$0")/../../../.."
OUT=research/glm53-flash-bringup-20260827/moe-loc-20260831/receipts
fails=0
SUITES_ALL="glm5_moe_loc_doors_gpu glm5_verify_batch_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu glm5_moe_epilogue_gpu glm5_mtp_head_gpu glm5_kpool_indexer_gpu glm5_matvec_doors_gpu hyper_connections_gpu hc_fused_pre_gpu hc_decode_ws_gpu mla_decode_split_gpu kda_fixture_gpu kda_fused_proj_gpu kda_fused_proj_bf16_gpu kda_quant_operand_gpu mla_gpu_forward"
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
  run "compose-$s.log" MEMRA_MOE_VROWS_DEV_TABLES=1 MEMRA_GLM5_HTOD_DIET=1 MEMRA_MOE_VROWS_PACK=0 -- "$s"
done
for s in $SUITES_WALK; do
  run "stat-$s.log" MEMRA_MOE_VROWS_DEDUP_STAT=1 MEMRA_MOE_VROWS_DEV_TABLES=0 MEMRA_MOE_VROWS_PACK=0 -- "$s"
done
echo "=========================================================="
if [ "$fails" -eq 0 ]; then echo "moe-loc battery: ALL SUITES PASS"; else echo "moe-loc battery: $fails SUITE(S) FAILED"; fi
exit "$fails"
