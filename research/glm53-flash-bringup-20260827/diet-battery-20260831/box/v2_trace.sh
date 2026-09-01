#!/usr/bin/env bash
# VREST PHASE CELL V2 — phase receipt on the vrest head: MEMRA_GLM5_SPEC_TRACE=2,
# [glm5-phase-v] now carries vrest=(vffn=) — bank the vffn share so the slope
# prediction (9.46 -> ~3.5-3.9 ms/row) gets its real-artifact receipt. Arms:
# (a) dfl-k3-vr (batched + vrest pairs), comparable field-for-field with the
#     flip-reprice cell-2 K3 row (verify 69.72 / vrest 45.61);
# (b) dfl-k3-z0 seam arm (MEMRA_GLM5_VERIFY_BATCH=0): ONE flag restores the per-row
#     mixer walk AND the per-(token,expert) MoE class — must reproduce the ~91.1 ms
#     zctl K=3 round wall (flip-battery 96.47 verify / flip-reprice zctl 91.14 wall).
# SHARES-not-walls law: traced ms are shares, the flip table's walls are cell V3's.
set -uo pipefail
OUT=/root/out-diet/v2
TAGS=d00-code,d02-code,d06-prose,l3-A4630
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_K=3)
mkdir -p "$OUT"

echo "######## V2 BOOT dfl-k3-vr (trace=2, batched+pairs) ########"
/root/out-diet/serve.sh start v2-dfl-k3-vr "${DFL[@]}" DIET_PHASE=vrest MEMRA_GLM5_SPEC_TRACE=2 || exit 1
python3 /root/out-diet/run_pool.py sample --out "$OUT/dfl-k3-vr" || exit 1
/root/out-diet/serve.sh doors v2-dfl-k3-vr "${DFL[@]}" DIET_PHASE=vrest || exit 1
python3 /root/out-diet/run_pool.py cell --out "$OUT/dfl-k3-vr" --pool both --mode greedy \
  --max-tokens 256 --tags "$TAGS"
grep "\[glm5-phase-v\]" /root/out-diet/logs/boot-v2-dfl-k3-vr.log > "$OUT/phase-v-vr.log" || true
echo "phase bursts vr: $(wc -l < "$OUT/phase-v-vr.log")"

echo "######## V2 BOOT dfl-k3-z0 (trace=2, =0 SEAM: per-row mixer + per-(token,expert) MoE) ########"
/root/out-diet/serve.sh start v2-dfl-k3-z0 "${DFL[@]}" MEMRA_GLM5_VERIFY_BATCH=0 MEMRA_GLM5_SPEC_TRACE=2 || exit 1
python3 /root/out-diet/run_pool.py sample --out "$OUT/dfl-k3-z0" || exit 1
# seam receipt is INVERTED by construction — checked directly, not via doors():
zlog=/root/out-diet/logs/boot-v2-dfl-k3-z0.log
echo "seam-gate: perrow=$(grep -c 'verify walk PER-ROW' "$zlog") batched=$(grep -c 'verify walk BATCHED' "$zlog") vrows=$(grep -c '\[glm5-vrows\]' "$zlog")"
grep -q "verify walk PER-ROW" "$zlog" || { echo "V2_z0_SEAMGATE_FAIL: no PER-ROW line"; exit 1; }
[ "$(grep -c 'verify walk BATCHED' "$zlog")" -eq 0 ] || { echo "V2_z0_SEAMGATE_FAIL: BATCHED line on the =0 arm"; exit 1; }
[ "$(grep -c '\[glm5-vrows\]' "$zlog")" -eq 0 ] || { echo "V2_z0_SEAMGATE_FAIL: vrows line on the =0 arm"; exit 1; }
python3 /root/out-diet/run_pool.py cell --out "$OUT/dfl-k3-z0" --pool both --mode greedy \
  --max-tokens 256 --tags "$TAGS"
grep "\[glm5-phase-v\]" "$zlog" > "$OUT/phase-v-z0.log" || true
echo "phase bursts z0: $(wc -l < "$OUT/phase-v-z0.log")"

echo "=== V2 IDENTITY vr vs z0 (one seam, both classes — tapes must match) ==="
python3 /root/out-diet/run_pool.py compare --a "$OUT/dfl-k3-vr" --b "$OUT/dfl-k3-z0" || echo "V2_IDENTITY_DIVERGENCE — STOP"

/root/out-diet/serve.sh stop
python3 /root/out-diet/looplaw_screen.py "$OUT"/*/
echo "V2_DONE"
