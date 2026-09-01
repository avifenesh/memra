#!/usr/bin/env bash
# CELL 3a — THE DECISIVE RECEIPT: does the glm5 spec route engage on PP4 or not?
#
# WHY THIS CELL EXISTS. Cell 3 set the full ship env (DFlash2 @ b33c0347, MEMRA_GLM5_SPEC=1,
# PMIN=0.7) on the PP4 1M posture. The boot announced "[glm5-spec] serve route ARMED" and
# "[spec-gate] ... spec-admission=on", and then the verify walk NEVER RAN: 0 [glm5-acc],
# 0 "verify walk BATCHED per layer", 0 door T/X/K/W announces, 0 PMIN line, and decode
# 23.06-23.34 tok/s against the PLAIN boot's 23.34 - identical. Two candidate causes were
# differenced and BOTH REFUTED by measurement (receipts/c3diag/):
#   MEMRA_REUSE_POOL=0 (this window's own addition)  -> reverting it changed nothing
#   auto-K choosing K=0 at concurrency 1             -> an operator pin MEMRA_SPEC_K=3 was
#                                                       honored ("[spec-k] operator pin K=3")
#                                                       and STILL nothing engaged
# The code says why (worker.rs glm5_sharded_placement_admits):
#     (2..=3).contains(&fence_stages) && !tp_set(step_tp) && !tp_set(step_ep)
# with the reason stated in its own doc comment: the verify-walk ppN twin, per-stage rollback
# and last-stage MTP chain are red-proven by glm5-spec-ppn-gate at stages=2 and stages=3
# only, and "a stage count outside that set has NO gate receipt and refuses ... fail-closed
# is the default, extended deliberately, never inferred."
#
# So this is a DELIBERATE fail-closed refusal, not a bug - and it collides with the 1M
# posture, because the only demonstrated 1M config is PP4 (4 stages; a 1M context needs all
# four cards). This cell turns the code reading into a MEASUREMENT: same binary, same spec
# env, same MEMRA_CTX, ONE variable = the PP stage count.
#
#   arm ppab-pp4   MEMRA_PP_STAGES=4 SPLITS=13,26,39 devices 0,1,2,3   expect: spec REFUSED
#   arm ppab-pp3   MEMRA_PP_STAGES=3 SPLITS=15,30    devices 0,1,2     expect: spec ENGAGES
#
# CTX is pinned to 131072 on BOTH arms so the stage count is the only difference (a 1M CTX
# would not fit three cards and would confound the comparison with a capacity refusal).
# UNTIMED: this is an engagement/identity cell, not a perf cell.
set -uo pipefail
OUT=/root/out-1m
D=$OUT/receipts/c3a-ppab
mkdir -p "$D"
SPEC=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7 MEMRA_SPEC_STATS=1)
{
date -u +%FT%TZ
echo "######## CELL 3a: PP4-vs-PP3 SPEC ENGAGEMENT A/B (untimed, one variable) ########"
for arm in pp4 pp3; do
  case $arm in
    pp4) PP=(CUDA_VISIBLE_DEVICES=0,1,2,3 MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PP_SPLITS=13,26,39) ;;
    pp3) PP=(CUDA_VISIBLE_DEVICES=0,1,2   MEMRA_PP_STAGES=3 MEMRA_PP_DEVICES=0,1,2   MEMRA_PP_SPLITS=15,30) ;;
  esac
  echo; echo "######## ARM $arm ########"
  bash "$OUT/serve.sh" start "ppab-$arm" "${SPEC[@]}" "${PP[@]}" MEMRA_CTX=131072 \
    || { echo "ARM $arm BOOTFAIL"; continue; }
  bash "$OUT/rung.sh" "ppab-$arm" "A16K" 64400 128 greedy "$D/$arm" || echo "ARM $arm rung failed"
  echo "--- SPEC ENGAGEMENT VERDICT for $arm ---"
  L=$OUT/logs/boot-ppab-$arm.log
  tot=0
  for pat in "\[glm5-acc\]" "verify walk BATCHED per layer" "\[glm5-vrows\]" "PMIN=0.700" \
             "\[bf16-tcols-wide\] engaged" "\[bf16-tcols-x1\] engaged" \
             "\[topk-shards\] engaged" "\[glm5-verify-ws\] engaged"; do
    n=$(grep -c -- "$pat" "$L"); tot=$((tot+n))
    printf "  %-38s %s\n" "$pat" "$n"
  done
  echo "  SPEC EVIDENCE TOTAL for $arm = $tot   (0 = the route refused and served plain)"
  echo "  stage fence lines:"; grep -E "cross-device transport" "$L" | head -1
  bash "$OUT/serve.sh" stop
done
echo
echo "=== SIDE BY SIDE (decode tok/s at the same 16k prompt, same binary, same spec env) ==="
python3 - "$D" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for arm in ("pp4", "pp3"):
    p = root / arm / "A16K-greedy.json"
    if not p.exists():
        print(f"  {arm}: MISSING"); continue
    j = json.load(open(p))
    u = j.get("usage") or {}
    print(f"  {arm}: pt={u.get('prompt_tokens')} TTFD={j['prefill_s']}s "
          f"prefill={j['prefill_tok_s']} decode_span={j['decode_tok_s']} "
          f"steady_p50={j['decode_steady_tok_s']} ct={u.get('completion_tokens')}")
PY
date -u +%FT%TZ
echo "C3A_DONE"
} 2>&1 | tee "$OUT/logs/c3a.log"
