#!/usr/bin/env bash
# Watchdog unit test: a real-style nvidia-smi header, rows at 93 C / 97 % fan; expect THERMAL-STOP after 3 ten-second
# ticks (about 30 s) and the marked sleep killed. Then a 96 C row: immediate stop.
set -u
W=/home/avifenesh/.cache/memra-maxq/kv-vmm/scratch/selftest/wd
rm -rf /home/avifenesh/.cache/memra-maxq/kv-vmm/scratch/selftest/wd
mkdir -p "$W"
WD=/home/avifenesh/.cache/memra-maxq/kv-vmm/kit/watchdog.py
for case in "93 97 %" "96 40 %"; do
  set -- $case
  csv=$W/t$1.csv
  printf 'timestamp, index, temperature.gpu, fan.speed [%%], memory.used [MiB]\n' > "$csv"
  ( for i in $(seq 1 240); do echo "2026/11/04 22:00:00.000, 2, $1, $2 %, 3 MiB" >> "$csv"; sleep 0.25; done ) &
  feeder=$!
  env KIT_RUN_ID=wdtest-$1 sleep 120 &
  sp=$!
  start=$(date +%s)
  python3 "$WD" --csv "$csv" --deadline $((start + 600)) --marker "KIT_RUN_ID=wdtest-$1" --log "$W/log" --stopfile "$W/stop"
  echo "case $1 C fan $2: watchdog rc=$? after $(( $(date +%s) - start )) s"
  if kill -0 "$sp" 2>/dev/null; then echo "  marked sleep STILL ALIVE"; kill "$sp"; else echo "  marked sleep stopped"; fi
  kill "$feeder" 2>/dev/null
  wait 2>/dev/null
done
cat "$W/stop" "$W/log"
