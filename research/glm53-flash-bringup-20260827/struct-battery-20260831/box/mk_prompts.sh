#!/usr/bin/env bash
# Materialize the house prompt pools as one-file-per-prompt dirs for the engine probe
# (cell 5). Same pools as every banked baseline (tp2-battery twin of this script).
set -euo pipefail
OUT=${OUT:-/root/out-struct}
DECODE_POOL=/root/memra-struct/research/glm53-flash-bringup-20260827/decode-attribution-receipts/prompts.json
L3_POOL=/root/l3-ab-prompts.json
python3 - "$OUT" "$DECODE_POOL" "$L3_POOL" <<'EOF'
import json, os, sys
out, dp, l3 = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(dp))
os.makedirs(f"{out}/prompts-decode", exist_ok=True)
for p in d["decode"]:
    open(f"{out}/prompts-decode/d{p['idx']:02d}-{p['kind']}.txt", "w").write(p["text"])
print("decode pool:", len(d["decode"]), "prompts")
l = json.load(open(l3))
# cell-5 identity subset: first 4 decode prompts + WARM (tape cost is 200 tok/prompt)
os.makedirs(f"{out}/prompts-c1", exist_ok=True)
for p in d["decode"][:4]:
    open(f"{out}/prompts-c1/d{p['idx']:02d}-{p['kind']}.txt", "w").write(p["text"])
open(f"{out}/prompts-c1/l3-WARM.txt", "w").write(l["WARM"])
print("c1 identity subset ready")
EOF
