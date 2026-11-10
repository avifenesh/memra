#!/usr/bin/env bash
# Rig, inside the caller's gpulock lease: start a timed RTX 5090 cell from a cool card and sample
# the card at 1 Hz while it runs (temperature, SM clock, power, the software thermal slowdown
# flag). Waits (at most 8 min) for the card to read at most 62 C, then runs CMD.
# usage: cool5090.sh SAMPLES.csv CMD...
set -u
CSV=$1; shift
Q=timestamp,temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown
waitfor --until '[ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 62 ]' --every 5s --timeout 8m --quiet > /dev/null 2>&1
echo "# start $(date -Is) temp $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C" > "$CSV"
nvidia-smi --query-gpu=$Q --format=csv,noheader -l 1 >> "$CSV" 2>&1 &
smi=$!
"$@"
rc=$?
kill $smi 2>/dev/null
wait $smi 2>/dev/null
echo "# end $(date -Is) rc=$rc max_temp $(grep -v '^#' "$CSV" | awk -F', ' '{if ($2+0 > m) m = $2+0} END {print m}') C sw_thermal_slowdown_samples $(grep -v '^#' "$CSV" | awk -F', ' '$5 == "Active"' | wc -l) of $(grep -vc '^#' "$CSV")" >> "$CSV"
exit $rc
