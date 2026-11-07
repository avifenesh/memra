#!/usr/bin/env bash
# Runs ON unit2 under the lease: the exactness battery on the glue-follow-up binary (k232) against
# the #1089 binary (gdn6), every lever of both lanes rolled back as the off arm, then one cold 131k
# served prefill of k232 under Nsight Systems.
set -u
D=$HOME/.cache/memra-gb10-next
M=$(ls -d "$D"/bins/gdn6-* | tail -n 1)
L=$(ls -d "$D"/bins/k232-* | tail -n 1)
R=$D/runs/glue3-gates
mkdir -p "$R"
echo "base=$M lane=$L" > "$R/chain.txt"
OFF_ENV="MEMRA_GDN_TRAFFIC=0 MEMRA_GDN_K45=0 MEMRA_PRIME_NORM_FUSE=0 MEMRA_DP4A_TILED=0 MEMRA_FFN_GATEUP_FUSE=0 MEMRA_PRIME_RESID_EPI=0 MEMRA_GDN_K2_32=0" \
  bash "$D/box/gates_next3.sh" "$L" "$M" "$R/gates" "$D/tree" > "$R/gates.out" 2>&1
echo "gates done" >> "$R/chain.txt"
OUT=$D/runs/profile-glue3-131k
mkdir -p "$OUT"
bash "$D/box/profile_arm.sh" "$OUT" "$L" "" 131072 > "$D/logs/profile-glue3.out" 2>&1
TTFT=$(python3 -c "import json;print([json.loads(l) for l in open('$OUT/pf.jsonl') if l.strip()][-1]['ttft_s'])")
python3 "$D/box/parts_last.py" "$OUT/prof.sqlite" "$TTFT" --json "$OUT/parts-last.json" > "$OUT/parts-last.txt" 2>&1
python3 "$D/box/shapes_last.py" "$OUT/prof.sqlite" "$TTFT" gdn_ ssm_conv l2_norm add_f32 rms_norm add_rms_norm quantize sigmoid sig_mul mul_f32 q_gate q_split gated_rmsnorm qmatvec mul_mat silu rope > "$OUT/shapes.txt" 2>&1
echo "profile ttft=$TTFT" >> "$R/chain.txt"
echo CHAIN11-DONE >> "$R/chain.txt"
