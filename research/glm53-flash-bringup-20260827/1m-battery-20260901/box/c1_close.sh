#!/usr/bin/env bash
# CELL 1 CLOSE, called the moment R1M-greedy lands.
#
# BUDGET DECISION, stated for the record: the cell-1 script would next run the 1M PLAIN
# vendor-default twin (a second full ~1M prime). That twin is DROPPED and its prime budget
# is reallocated to the SHIP-CONFIG 1M rung (cell 3, greedy + vendor). Reason: the owner's
# question is "today's tok/s at 1M on the SHIP config", so the vendor-default row that a
# serving claim actually requires is the SHIP one, not the plain baseline's. The demo's
# plain 1M greedy/sampled pair already agreed to 0.01% on prefill and 0.6% on decode, and
# the prime rate is cross-checked here by the whole depth ladder being rate-flat.
set -uo pipefail
OUT=/root/out-1m
D=$OUT/receipts/c1
# 1) stop the cell-1 runner and its probe BEFORE the vendor twin can start a second prime.
#    PID-scoped: match the runner by its own script path, verify /proc, never a bare pkill.
for pid in $(pgrep -f "bash /root/out-1m/c1_prime.sh" 2>/dev/null; pgrep -f "bash c1_prime.sh" 2>/dev/null); do
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *c1_prime.sh*) echo "[c1close] killing runner pid=$pid cmd=$cmd"; kill "$pid" 2>/dev/null ;;
    *) echo "[c1close] skip pid=$pid cmd=$cmd" ;;
  esac
done
for pid in $(pgrep -f "probe.py R1M .* vendor" 2>/dev/null); do
  echo "[c1close] killing a started vendor probe pid=$pid"; kill "$pid" 2>/dev/null
done
for pid in $(pgrep -f "vramwatch.sh $D/vram.csv" 2>/dev/null); do
  echo "[c1close] stopping vramwatch pid=$pid"; kill "$pid" 2>/dev/null
done
sleep 2
echo "=== PER-CARD VRAM PEAK over cell 1 (cards are 97,887 MiB) ==="
python3 - "$D/vram.csv" <<'PY' | tee "$D/vram-peaks.txt"
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if r and r[0] != "ts"]
print(f"samples={len(rows)} first={rows[0][0]} last={rows[-1][0]}")
for g in range(4):
    pk = max(int(r[g+1]) for r in rows)
    print(f"  gpu{g}: peak {pk} MiB ({100*pk/97887:.1f}% of the card)")
print("  demo phase7 peaks: 81945 / 80121 / 80089 / 94905 MiB")
PY
echo "=== boot-wide error census (must be 0) ==="
grep -cE "out.of.memory|panicked|CUDA_ERROR|engine-error|OUT_OF_MEMORY|\[admit-oom\]" \
  "$OUT/logs/boot-c1-plain.log" | tee "$D/error-census.txt"
echo "=== loop-law screen ==="
python3 "$OUT/looplaw_screen.py" "$D" | tee "$D/looplaw.txt"
bash "$OUT/serve.sh" stop
echo "C1_CLOSED $(date -u +%FT%TZ)"
