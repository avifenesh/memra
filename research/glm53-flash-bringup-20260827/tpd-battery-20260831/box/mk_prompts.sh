#!/usr/bin/env bash
# Materialize the house prompt pools as one-file-per-prompt dirs for the engine probe.
# Same pools as every banked baseline (tp2-battery / struct-battery twin of this script) so
# the diet rows are directly comparable to the banked v1 rows.
#   prompts-decode : the 10 decode-attribution real prompts (the pricing pool)
#   prompts-l3     : WARM (~0.4k) + A4630 (4626 tok, the "3.7k" TTFT depth row)
#   prompts-c1     : cell-1 identity subset = first 4 decode prompts + WARM
#   prompts-tiny   : the 12-token tiny-prime pool (keeps prime in the exact small-t regime)
set -euo pipefail
OUT=${OUT:-/root/out-tpd}
DECODE_POOL=/root/memra-tpd/research/glm53-flash-bringup-20260827/decode-attribution-receipts/prompts.json
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
os.makedirs(f"{out}/prompts-l3", exist_ok=True)
for k in ("WARM", "A4630"):
    open(f"{out}/prompts-l3/l3-{k}.txt", "w").write(l[k])
    print("l3", k, len(l[k]), "chars")
os.makedirs(f"{out}/prompts-c1", exist_ok=True)
for p in d["decode"][:4]:
    open(f"{out}/prompts-c1/d{p['idx']:02d}-{p['kind']}.txt", "w").write(p["text"])
open(f"{out}/prompts-c1/l3-WARM.txt", "w").write(l["WARM"])
print("c1 subset ready")
os.makedirs(f"{out}/prompts-tiny", exist_ok=True)
# byte-identical to the banked tp2-battery prompts-tiny/t0.txt
# (46 bytes, sha256 de11681964f01762b7d78110ec332bc0fc74bbf96ba83cce0136f326adceb02b)
open(f"{out}/prompts-tiny/t0.txt", "w").write(
    "Explain in one paragraph why the sky is blue.\n"
)
print("tiny pool ready")
EOF
sha256sum "$OUT"/prompts-tiny/t0.txt
wc -c "$OUT"/prompts-c1/*.txt "$OUT"/prompts-l3/*.txt
