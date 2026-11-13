#!/usr/bin/env bash
# Rig, inside a gpulock lease on the RTX 5090 (memra#1266): one boot of the normal-concurrency
# cell #1173 and #1255 used (research/drafter-admission-20261112/harness/normal_cell.py: closed-loop
# clients at c = 1, 2, 4, 8, 6 requests per client, greedy chat, 160 new tokens, streamed; ctx
# 8,192, 8 sessions) on Qwen3.8-27B NVFP4 + Q5_K, on the MTP route this change acts on: the masked
# 32,768-row MTP head, no DFlash drafter. The power limit is recorded, never changed.
# usage: normal_5090.sh CAMPAIGN NAME BIN_PATH   -> <lane>/runs/CAMPAIGN/NAME
set -u
D=<lane>
CAMP=$1; NAME=$2; BIN=$3
CD=$D/runs/$CAMP; OUT=$CD/$NAME; mkdir -p "$OUT"
M=<models>/q38-gguf
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
{ date -Is; echo "$NAME"; sha256sum "$BIN" | sed "s|$HOME|~|"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader
  echo "ac=$(cat /sys/class/power_supply/ADP0/online) nv_tgp=$(cat /sys/class/firmware-attributes/asus-armoury/attributes/nv_tgp/current_value) ppt_pl1_spl=$(cat /sys/class/firmware-attributes/asus-armoury/attributes/ppt_pl1_spl/current_value) profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unreadable)"
  uptime; } > "$OUT/guard.log"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,enforced.power.limit,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown --format=csv,noheader -l 1 > "$OUT/telemetry.csv" 2>/dev/null &
TP=$!
export CELL_MODEL=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf CELL_PORT=18266
export PLAN=1,2,4,8 REQS=6 MAXTOK=160 STREAM=1 TAG=$NAME
export EXTRA_ENV="MEMRA_MTP_DRAFT=$M/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"
setsid python3 "$D/box/normal_cell.py" "$OUT" "$BIN" "$NAME" > "$OUT/cell.log" 2>&1 &
CP=$!
if ! waitfor --gone "$CP" --timeout 20m --quiet; then
  echo "$(date -Is) cap 20 min reached; stopping group $CP" >> "$OUT/guard.log"; kill -TERM -- -"$CP"
  waitfor --gone "$CP" --timeout 2m --quiet || kill -KILL -- -"$CP"
fi
wait "$CP"; rc=$?
kill "$TP" 2>/dev/null
echo "cell $NAME rc=$rc $(date -Is)" >> "$OUT/guard.log"
echo "profile_end=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unreadable) $(date -Is)" >> "$OUT/guard.log"
{ for pat in 'out of memory' 'NOT parked' 'pooled MTP verify capped' 'pre-cohort reclaim' 'graph replay suspended' 'spec-pool] pooled MTP cohort engaged'; do
    echo "$pat=$(grep -a -c -F "$pat" "$OUT/server-$NAME.log")"; done; } > "$OUT/summary.txt"
gzip -f "$OUT/server-$NAME.log" "$OUT/telemetry.csv"
echo "$(date -Is) NORMAL-DONE $NAME rc=$rc" >> "$CD/campaign.log"
exit $rc
