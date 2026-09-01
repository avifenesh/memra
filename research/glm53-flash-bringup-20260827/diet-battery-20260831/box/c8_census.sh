#!/usr/bin/env bash
# DECODE-DIET CELL 8 — launch/alloc/sync census re-run on the WINNER arm (feeds the
# matvec-pass lane decision: where do the remaining ms sit?). Uses the launch-diet
# census instrument (nsys 2026.1.3 if installed, else the launch-econ constant alone)
# with PP_ENV = the serving placement env + the winner's door flags, per the LANE.md
# box-protocol section. The census's own MEMRA_CTX=8192/MAX_SESSIONS=1/FUSED_EPI=1 pins
# are the instrument's config (box A comparability), recorded in its provenance receipt.
# Usage: c8_census.sh "<door flags, space separated>"  (empty string = doors-off census)
set -uo pipefail
WINNER_DOORS="${1:?pass the winner door flags (or '' for none)}"
OUT=/root/out-diet/c8
mkdir -p "$OUT/prompts"
cd "$OUT"
# real prompt text for the census (its /root/prompts fallback is absent on this box):
# the decode pool's first code prompt, the same real-prompts law every cell obeys.
python3 - <<'PY'
import json
d = json.load(open("/root/memra/research/glm53-flash-bringup-20260827/decode-attribution-receipts/prompts.json"))
open("prompts/p0.txt", "w").write(d["decode"][0]["text"])
PY

export CUDA_VISIBLE_DEVICES=0,1,2
export NVIDIA_TF32_OVERRIDE=0
MODEL_DIR=/root/models/glm53-nvfp4 \
BIN=/root/memra-diet/target/release/memra-server \
PORT=18412 OUT=diet-census \
PP_ENV="MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1 MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 $WINNER_DOORS" \
bash /root/memra-diet/research/glm53-flash-bringup-20260827/launch-diet-20260830/census-decode-phases.sh
