#!/usr/bin/env bash
# Runs ON a GB10 host beside a cell: the GPU's clock-event reasons and the CPU's busiest processes,
# to attribute a clock drop. usage: clockmon.sh OUTDIR SECONDS
set -u
OUT=$1; SECS=$2
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_power_brake_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown \
  --format=csv,noheader -lms 500 > "$OUT/clock-reasons.csv" 2>&1 &
NP=$!
end=$(( $(date +%s) + SECS ))
while [ "$(date +%s)" -lt "$end" ]; do
  { date -Is; cat /proc/loadavg; top -b -n 1 -o %CPU | sed -n 8,14p; } >> "$OUT/cpu.log"
  timeout 10 tail -f /dev/null 2>/dev/null
done
kill "$NP"
