#!/bin/bash
# A stand-in for dsv41-spec in a dry run of the lane's scripts: reads --plan and --out like the binary and prints one
# JSON line a job in the binary's shape. STANDIN_RED names a job id to fail (then, with --stop-on-fail, the run stops).
plan=""; out=""; stop=0
while [ $# -gt 0 ]; do
  case "$1" in --plan) plan=$2; shift ;; --out) out=$2; shift ;; --stop-on-fail) stop=1 ;; esac
  shift
done
echo "[dsv41-tp] ranks=stand-in transport=${MEMRA_TP_TRANSPORT:-unset} (stand-in)" >&2
[ -f "$plan" ] || { echo "stand-in: no plan $plan" >&2; exit 2; }
mkdir -p "$out"
failed=0; n=0
while read -r job id rest; do
  [ -z "$job" ] && continue
  n=$((n + 1))
  if [ $stop = 1 ] && [ $failed -gt 0 ]; then echo "{\"stopped\":true,\"skipped\":[\"$id\"]}"; break; fi
  ok=true; bad=null
  if [ "$id" = "${STANDIN_RED:-}" ]; then ok=false; bad='"sequence 2 step 17 (row 2 of 8, position 61)"'; failed=$((failed + 1)); fi
  case "$job" in
    batch-rows) echo "{\"job\":\"batch-rows\",\"id\":\"$id\",\"pass\":$ok,\"sequences\":8,\"rows\":503,\"logit_rows_equal\":503,\"tap_rows_equal\":503,\"device_argmax_equal\":503,\"calls_by_width\":[0,1,2],\"first_bad\":$bad,\"prompts\":[1],\"steps\":[1],\"arm_a_s\":1.0,\"arm_b_s\":1.0}" ;;
    batch-time) echo "{\"job\":\"batch-time\",\"id\":\"$id\",\"pass\":$ok,\"argmax_1_ms\":12.3,\"argmax_2_ms\":14.0,\"argmax_4_ms\":16.1,\"argmax_8_ms\":20.9,\"fwd_1_ms\":11.0,\"fwd_2_ms\":12.5,\"fwd_4_ms\":14.9,\"fwd_8_ms\":19.0,\"batched_1_ms\":12.4,\"batched_2_ms\":14.1,\"batched_4_ms\":16.3,\"batched_8_ms\":21.2,\"alone_1_ms\":12.3,\"alone_2_ms\":24.6,\"alone_4_ms\":49.2,\"alone_8_ms\":98.4,\"distinct_experts_per_layer_mean\":28.8,\"reps\":24}" ;;
    *) echo "{\"job\":\"$job\",\"id\":\"$id\",\"pass\":$ok}" ;;
  esac
done < "$plan"
echo "SPEC-DONE jobs $n failed $failed"
[ $failed -eq 0 ]
