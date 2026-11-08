#!/usr/bin/env bash
# Runs ON unit 1 under the lease: one cold 131k served prefill of the stack head (#1117) under nsys,
# then the request-window parts (PRE from the timestamped run: 0.446 s), API gaps and shapes.
set -u
D=$HOME/.cache/memra-gb10-r4
N=$(ls -d "$D"/bins/ns-* | tail -n 1)
OUT=$D/runs/profile-stack-131k
bash "$D/box/profile_arm.sh" "$OUT" "$N" "" 131072 > "$D/logs/profile-stack.out" 2>&1
TTFT=$(python3 -c 'import json,sys; print([json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]["ttft_s"])' "$OUT/pf.jsonl")
python3 "$D/box/parts_req.py" "$OUT/prof.sqlite" "$TTFT" 0.446 --json "$OUT/parts-req.json" > "$OUT/parts-req.txt" 2>&1
python3 "$D/box/gaps_api.py" "$OUT/prof.sqlite" "$TTFT" --pre 0.446 --min-ms 1 --show 20 > "$OUT/gaps-api-req.txt" 2>&1
python3 "$D/box/shapes_last.py" "$OUT/prof.sqlite" "$TTFT" fa_prefill_qw_h2p mul_mat_f8f4v3 f8f4v2_image_kernel > "$OUT/shapes.txt" 2>&1
echo PROFILE-STACK-DONE >> "$OUT/run.log"
