#!/usr/bin/env bash
# CELL 1 — boot gates both arms + correctness spot (untimed, exactness only).
# Bar: every concurrent greedy tape byte-identical to its solo tape on the SAME boot
# (cross-session contamination is the gate's red class), AND ON solo == OFF solo
# (served-path B=1 class pin: with one ready session the ON arm still walks the batched body).
set -uo pipefail
OUT=/root/out-hbatch/c1
RP="python3 /root/out-hbatch/run_pool.py"
mkdir -p "$OUT"
fail=0

# ---- OFF arm (today's default) ----
/root/out-hbatch/serve.sh start c1-off || exit 1
$RP sample --out "$OUT/off" || fail=1
$RP solo --picks 0,6,3,11 --out "$OUT/off/solo" || fail=1
$RP conc --n 2 --picks 0,6 --out "$OUT/off/conc-a" || fail=1
$RP conc --n 2 --picks 3,11 --out "$OUT/off/conc-b" || fail=1

# ---- ON arm ----
/root/out-hbatch/serve.sh start c1-on MEMRA_HYPER_BATCH=1 || exit 1
$RP sample --out "$OUT/on" || fail=1
$RP solo --picks 0,6,3,11 --out "$OUT/on/solo" || fail=1
$RP conc --n 2 --picks 0,6 --out "$OUT/on/conc-a" || fail=1
$RP conc --n 2 --picks 3,11 --out "$OUT/on/conc-b" || fail=1
# width spot at the ladder top on the ON boot: c=12 concurrent tapes vs solo (free coverage
# of the contamination class at real width; timing NOT read from this cell)
$RP solo --picks 1,2,4,5,7,8,9,10 --out "$OUT/on/solo" || fail=1
$RP conc --n 12 --out "$OUT/on/conc-w" || fail=1

/root/out-hbatch/serve.sh stop

echo "=== IDENTITY BARS ==="
$RP compare --a "$OUT/off/solo" --b "$OUT/off/conc-a" || fail=1
$RP compare --a "$OUT/off/solo" --b "$OUT/off/conc-b" || fail=1
$RP compare --a "$OUT/on/solo"  --b "$OUT/on/conc-a"  || fail=1
$RP compare --a "$OUT/on/solo"  --b "$OUT/on/conc-b"  || fail=1
$RP compare --a "$OUT/on/solo"  --b "$OUT/on/conc-w"  || fail=1
$RP compare --a "$OUT/off/solo" --b "$OUT/on/solo"    || fail=1
$RP compare --a "$OUT/off/conc-a" --b "$OUT/on/conc-a" || fail=1
$RP compare --a "$OUT/off/conc-b" --b "$OUT/on/conc-b" || fail=1

echo "=== LOOP-LAW SCREEN ==="
python3 /root/out-hbatch/looplaw_screen.py "$OUT"/off "$OUT"/on

[ "$fail" -eq 0 ] && echo "C1_VERDICT=GREEN" || echo "C1_VERDICT=RED"
exit "$fail"
