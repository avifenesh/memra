#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1266 review 2): the riders cell of memra#1244
# (riders_cell_1266.py: 2 greedy streams, then 4 greedy-with-penalty plain requests that ride the
# pooled MTP cohort's forwards on this card) on the GB10 launcher's model files, one boot. On the
# RTX 5090 a rider steps plain inside each round (no cost table for the model), so the memory
# cap's rider path runs here: the forced-cap arm (MEMRA_SPEC_POOL_SERIAL=1) must count capped
# rider rows. Correctness cell: texts and counters, no timing claim.
# usage: riders_gb10.sh CAMPAIGN NAME ARM [KEY=VALUE ...]   -> runs/CAMPAIGN/NAME/
set -u
D=<lane>
CAMP=$1; NAME=$2; BIN=$D/bins/$3/memra-server; shift 3
OUT=$D/runs/$CAMP/$NAME; mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "RIDERS-FAIL $NAME (no binary)" >> "$D/runs/$CAMP/campaign.log"; exit 1; }
G8=<models>/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
extra="MEMRA_MTP_DRAFT=$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf;MEMRA_PREFIX_STABLE_BOUNDARY=1"
for kv in "$@"; do extra="$extra;$kv"; done
{ date -Is; echo "$NAME extra: $*"; sha256sum "$BIN" | sed "s|$HOME|~|"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; uptime; } > "$OUT/guard.log"
CELL_MODEL=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf CELL_PORT=18266 CELLS=${CELLS:-2:4:pen} \
  EXTRA_ENV="$extra" TAG=$NAME timeout 1300 python3 "$D/box/riders_cell_1266.py" "$OUT" "$BIN" "$NAME" > "$OUT/cell.log" 2>&1
rc=$?
{ for pat in 'out of memory' 'NOT parked' 'pooled MTP verify capped' 'pre-cohort reclaim' 'spec-pool] pooled MTP cohort engaged'; do
    echo "$pat=$(grep -a -c -F "$pat" "$OUT/server-$NAME.log" 2>/dev/null)"; done; } > "$OUT/summary.txt"
gzip -f "$OUT/server-$NAME.log"
echo "$(date -Is) RIDERS-DONE $NAME rc=$rc" >> "$D/runs/$CAMP/campaign.log"
exit $rc
