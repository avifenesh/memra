#!/usr/bin/env bash
# Rig (RTX 5090 Laptop GPU, run inside the caller's gpulock lease), memra#1266: one boot of the
# 29-turn agentic replay on the MTP route, memra#1221 part 2's shape (rtx_agentic.sh): Qwen3.8-27B
# NVFP4+Q5_K with the masked 32,768-row MTP draft, ctx 32,768, 8 sessions, prefix cache 4,000 MB,
# no host tier, MEMRA_PREFIX_STABLE_BOUNDARY=1, MEMRA_TICK_TRACE=1; the driver's prime pass at
# max_tokens 1 then the timed pass at c = 2, thinking on, temperature 0, seed 1234, 768 tokens.
# No trim probe. The enforced power limit is recorded, never changed.
# usage: rtx_cell.sh CAMPAIGN NAME BIN_PATH [KEY=VALUE ...]   -> <lane>/runs/CAMPAIGN/NAME
set -u
D=<lane>
CAMP=$1; NAME=$2; BIN=$3; shift 3
CD=$D/runs/$CAMP; OUT=$CD/$NAME; mkdir -p "$OUT"
M=<models>/q38-gguf
PORT=8266
if ss -ltn | grep -q ":$PORT "; then echo "$(date -Is) $NAME port $PORT busy; refusing" >> "$CD/campaign.log"; exit 4; fi
{ date -Is; echo "$NAME extra: $*"; sha256sum "$BIN" "$D/box/ab-workload.json" | sed "s|$HOME|~|";
  nvidia-smi --query-gpu=name,driver_version,enforced.power.limit,temperature.gpu,memory.used --format=csv,noheader
  echo "ac=$(cat /sys/class/power_supply/ADP0/online) nv_tgp=$(cat /sys/class/firmware-attributes/asus-armoury/attributes/nv_tgp/current_value) ppt_pl1_spl=$(cat /sys/class/firmware-attributes/asus-armoury/attributes/ppt_pl1_spl/current_value) profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unreadable)"
  uptime; } > "$OUT/guard.log"
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
env MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:$PORT MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=8 MEMRA_KV_HOST_MB=0 \
  MEMRA_PREFIX_CACHE_MB=4000 MEMRA_MODELS="q38=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
  MEMRA_MTP_DRAFT="$M/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf" MEMRA_PREFIX_STABLE_BOUNDARY=1 MEMRA_TICK_TRACE=1 \
  "$@" setsid python3 "$D/box/tsrun.py" "$BIN" > "$OUT/server.log" 2>&1 &
sp=$!
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,utilization.gpu,memory.used,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown \
  --format=csv,noheader -l 1 > "$OUT/telemetry.csv" 2>/dev/null &
tp=$!
rc=0
if waitfor --until "curl -sf -o /dev/null http://127.0.0.1:$PORT/readyz && exit 0 || exit 1" --fail-if "! kill -0 $sp" --every 5s --timeout 8m > "$OUT/run.log" 2>&1; then
  CLIENT_MAX_TIME=600 BENCH_BASE=http://127.0.0.1:$PORT BENCH_MODEL=q38 timeout 900 \
    python3 "$D/box/agent_turns.py" "$D/box/ab-workload.json" "$OUT" "$NAME" --users "${USERS:-2}" --think 1 \
    --temperature 0 --seed 1234 --max-tokens 768 >> "$OUT/cells.log" 2>&1 || rc=$?
  curl -s http://127.0.0.1:$PORT/metrics > "$OUT/metrics.json" 2>/dev/null
else
  echo "boot FAILED" >> "$OUT/run.log"; rc=3
fi
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$OUT/compute-apps.txt" 2>&1
kill -TERM "$sp" 2>/dev/null
waitfor --gone "$sp" --timeout 2m >> "$OUT/run.log" 2>&1 || kill -KILL -- -"$sp" 2>/dev/null
kill "$tp" 2>/dev/null
{ for pat in 'out of memory' 'NOT parked' 'parked session back' 'graph replay suspended' 'step-OOM reclaim' \
             '[p1266] alloc-fail' '[p1266] pooled-verify-fail' 'reclaim-on-defer' '[prefix-cache] reserve' 'pooled MTP verify capped' 'pre-cohort reclaim'; do
    echo "$pat=$(grep -a -c -F "$pat" "$OUT/server.log")"
  done; } > "$OUT/summary.txt"
tail -n 1 "$OUT/cells.log" | cut -c1-400 | sed "s/^/$NAME /" >> "$CD/campaign.log"
echo "profile_end=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unreadable) $(date -Is)" >> "$OUT/guard.log"
gzip -f "$OUT/server.log" "$OUT/telemetry.csv"
[ $rc = 0 ] && echo "$(date -Is) CELL-DONE $NAME" >> "$CD/campaign.log"
exit $rc
