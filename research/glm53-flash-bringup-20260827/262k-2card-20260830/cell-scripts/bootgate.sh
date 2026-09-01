#!/bin/bash
# Boot gate for the 262k 2-card PINNED-RECIPE cell (pre-registered bar 1):
#   BOTH PP devs must log a resident-experts decision ending in RESIDENT.
#   Any -> SLRU decision, or a FATAL/panic before ready, IS the cell result: bank and stop.
# Banks: the decision lines, the pp transport line, prefix-cache line, VRAM at ready.
# usage: bootgate.sh <server-log> <outdir>
set -u
SLOG=$1; OUT=$2
mkdir -p "$OUT"
{
  echo "# boot gate @ $(date -u +%FT%TZ)"
  grep -E "resident-experts decision" "$SLOG"
  grep -E "cross-device transport" "$SLOG" | head -2
  grep -iE "prefix-cache" "$SLOG" | head -3
  grep -E "listening on" "$SLOG"
} | tee "$OUT/bootgate-lines.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee "$OUT/vram-at-ready.csv"

FAIL=0
for dev in "PP dev0" "PP dev1"; do
  L=$(grep -F "resident-experts decision ($dev)" "$SLOG" | tail -1)
  if [ -z "$L" ]; then echo "GATE FAIL: no resident-experts decision line for $dev"; FAIL=1
  elif ! echo "$L" | grep -q -- "-> RESIDENT"; then echo "GATE FAIL: $dev decision is not RESIDENT: $L"; FAIL=1
  fi
done
grep -q -- "-> SLRU" "$SLOG" && { echo "GATE FAIL: an SLRU decision appears in the log"; FAIL=1; }
# case-SENSITIVE: the benign gpu-watch config line says "fatal Xid [..]" in lowercase;
# real failures log FATAL (e.g. the s23 arm's "FATAL load OOM after last slab") or panicked.
grep -qE "FATAL|panicked" "$SLOG" && { echo "GATE FAIL: FATAL/panic in boot log"; FAIL=1; }
grep -q "listening on" "$SLOG" || { echo "GATE FAIL: server never listened"; FAIL=1; }
if [ $FAIL -eq 0 ]; then echo "BOOT GATE PASS: both PP devs RESIDENT, no SLRU, no FATAL, listening"; fi
exit $FAIL
