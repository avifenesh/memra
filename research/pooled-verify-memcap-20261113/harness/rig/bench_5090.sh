#!/usr/bin/env bash
# Rig, inside a gpulock lease on the RTX 5090 (memra#1266): what a capped round costs. The
# decode-batch-bench --pooled timer (one session per entry, entry = its verify rows, depth --ctx +
# 37 i; serial = the per-session verifies back to back, pooled = one forward; bitwise pin first)
# at the cohort shapes the cap splits: c=2 K5+K5 (6,6), K5+K2 (6,3), K2+K2 (3,3); c=4 K2 x4
# (3,3,3,3) and two pairs (3,3); K3 x4 (4,4,4,4) and (4,4). Depth 4,096, 20 reps.
# usage: bench_5090.sh OUT_DIR
set -u
OUT=$1; mkdir -p "$OUT"
B=<lane>/bins5090/bench/decode-batch-bench
M=<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
{ date -Is; sha256sum "$B" | sed "s|$HOME|~|"; nvidia-smi --query-gpu=name,driver_version,enforced.power.limit,temperature.gpu --format=csv,noheader
  echo "ac=$(cat /sys/class/power_supply/ADP0/online) nv_tgp=$(cat /sys/class/firmware-attributes/asus-armoury/attributes/nv_tgp/current_value) ppt_pl1_spl=$(cat /sys/class/firmware-attributes/asus-armoury/attributes/ppt_pl1_spl/current_value) profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unreadable)"; } > "$OUT/guard.log"
# 1 Hz telemetry with the thermal throttle reasons (the timed-take rule reads it).
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,utilization.gpu,memory.used,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown \
  --format=csv,noheader -l 1 > "$OUT/telemetry.csv" 2>/dev/null &
tp=$!
for shape in 6,6 6,3 3,3 3,3,3,3 4,4,4,4 4,4; do
  timeout 300 "$B" "$M" --pooled "$shape" --ctx 4096 --reps 20 >> "$OUT/bench.log" 2>&1
  echo "shape $shape rc=$?" >> "$OUT/bench.log"
done
kill "$tp" 2>/dev/null; wait "$tp" 2>/dev/null; gzip -f "$OUT/telemetry.csv"
echo "profile_end=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unreadable) $(date -Is)" >> "$OUT/guard.log"
grep -E 'POOLED-TIME|BITWISE|MISMATCH|rc=' "$OUT/bench.log" > "$OUT/summary.txt"
