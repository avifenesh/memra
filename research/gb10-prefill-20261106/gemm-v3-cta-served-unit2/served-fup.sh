#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: one served cold-TTFT campaign that carries both
#   - ledger row 43, the unit 2 confirmation of #1102: B = the landed binary with MEMRA_MMQ_F8F4_V3=0
#     (the v2 mainloop), L = the landed binary (v3 for the plain and residual GEMMs);
#   - ledger row 45, the follow-up against main: N = the follow-up binary (CTA-form v3, gate/up on v3)
#     against L.
# Cells B L N N L B: B against L and N against L are each balanced in both orders. One fresh boot per
# cell, one cold prompt per length (32k / 131k / 260k) per boot.
# usage: served-fup.sh LANDED_BIN FOLLOWUP_BIN OUT
set -u
LB=$1; NB=$2; OUT=$3
D=$HOME/.cache/memra-f8f4v3
export TMPDIR=$D/tmp PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
mkdir -p "$OUT"
(cd "$LB" && sha256sum -c SHA256SUMS) > "$OUT/landed-bin.check" 2>&1 || { echo "LANDED-BIN-BAD"; exit 1; }
(cd "$NB" && sha256sum -c SHA256SUMS) > "$OUT/followup-bin.check" 2>&1 || { echo "FOLLOWUP-BIN-BAD"; exit 1; }
cp "$LB/COMMIT" "$OUT/landed-COMMIT"; cp "$NB/COMMIT" "$OUT/followup-COMMIT"
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader > "$OUT/clocks-start.txt"
bash "$D/h/campaign.sh" "$OUT" "32768 131072 260000" "1 1 1" \
  "B|$LB|MEMRA_MMQ_F8F4_V3=0" "L|$LB|" "N|$NB|" "N|$NB|" "L|$LB|" "B|$LB|MEMRA_MMQ_F8F4_V3=0" > "$OUT/campaign.out" 2>&1
python3 "$D/h/summarize.py" "$OUT" > "$OUT/summary.txt" 2>&1
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader > "$OUT/clocks-end.txt"
echo SERVED-FUP-DONE | tee -a "$OUT/summary.txt"
