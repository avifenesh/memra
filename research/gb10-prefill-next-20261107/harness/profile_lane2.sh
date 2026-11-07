#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: one cold 131k served prefill of the glue-2 binary under
# Nsight Systems, then the last-prefill per-part summary and the launch shapes.
set -u
D=$HOME/.cache/memra-gb10-next
B=$(ls -d "$D"/bins/glue2-* | tail -n 1)
OUT=$D/runs/profile-glue2-131k
mkdir -p "$OUT"
bash "$D/box/profile_arm.sh" "$OUT" "$B" "" 131072 > "$D/logs/profile-glue2.out" 2>&1
TTFT=$(python3 -c "import json;print([json.loads(l) for l in open('$OUT/pf.jsonl') if l.strip()][-1]['ttft_s'])")
python3 "$D/box/parts_last.py" "$OUT/prof.sqlite" "$TTFT" --json "$OUT/parts-last.json" > "$OUT/parts-last.txt" 2>&1
python3 "$D/box/shapes_last.py" "$OUT/prof.sqlite" "$TTFT" gdn_ ssm_conv l2_norm add_f32 rms_norm add_rms_norm quantize sigmoid sig_mul mul_f32 q_gate gated_rmsnorm qmatvec mul_mat silu rope > "$OUT/shapes.txt" 2>&1
echo "PROFILE-GLUE2-DONE ttft=$TTFT" >> "$OUT/run.log"
