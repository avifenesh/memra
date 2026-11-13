#!/usr/bin/env bash
# Rig, inside a gpulock lease on the RTX 5090 (memra#1266 review 2): the riders cell of memra#1244
# (`riders_cell_1266.py`: greedy streams, then greedy-with-penalty plain requests that ride the
# pooled MTP cohort) on Qwen3.8-27B NVFP4 + Q5_K with the masked MTP head, one boot. The
# forced-cap arm (MEMRA_SPEC_POOL_SERIAL=1) must count capped rider rows; the riders' texts are
# compared across arms. Correctness cell: no timing claim.
# usage: rider_5090.sh CAMPAIGN NAME BIN_PATH [KEY=VALUE ...]  -> <lane>/runs/CAMPAIGN/NAME
set -u
D=<lane>
CAMP=$1; NAME=$2; BIN=$3; shift 3
OUT=$D/runs/$CAMP/$NAME; mkdir -p "$OUT"
M=<models>/q38-gguf
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
extra="MEMRA_MTP_DRAFT=$M/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"
for kv in "$@"; do extra="$extra;$kv"; done
{ date -Is; echo "$NAME extra: $*"; sha256sum "$BIN" | sed "s|$HOME|~|"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,enforced.power.limit,memory.used --format=csv,noheader
  echo "ac=$(cat /sys/class/power_supply/ADP0/online) nv_tgp=$(cat /sys/class/firmware-attributes/asus-armoury/attributes/nv_tgp/current_value) ppt_pl1_spl=$(cat /sys/class/firmware-attributes/asus-armoury/attributes/ppt_pl1_spl/current_value) profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unreadable)"
  uptime; } > "$OUT/guard.log"
CELL_MODEL=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf CELL_PORT=18266 CELLS=${CELLS:-1:2:pen,2:1:pen} \
  EXTRA_ENV="$extra" TAG=$NAME timeout 1200 python3 "$D/tools/riders_cell_1266.py" "$OUT" "$BIN" "$NAME" > "$OUT/cell.log" 2>&1
rc=$?
echo "profile_end=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unreadable) $(date -Is)" >> "$OUT/guard.log"
{ for pat in 'out of memory' 'NOT parked' 'pooled MTP verify capped' 'pre-cohort reclaim' 'spec-pool] pooled MTP cohort engaged'; do
    echo "$pat=$(grep -a -c -F "$pat" "$OUT/server-$NAME.log" 2>/dev/null)"; done; } > "$OUT/summary.txt"
gzip -f "$OUT/server-$NAME.log"
echo "$(date -Is) RIDERS-DONE $NAME rc=$rc" >> "$D/runs/$CAMP/campaign.log"
exit $rc
