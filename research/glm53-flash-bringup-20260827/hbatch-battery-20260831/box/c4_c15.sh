#!/usr/bin/env bash
# CELL 2b — c=15 rung (the derived cap width, single batched chunk), TIMED, marker up,
# interleaved x3 per arm. Picks = all 14 pool items + a duplicate of d00 (named deviation:
# the pool has 14 items; the duplicate's tape is excluded from identity bars, timing only).
set -uo pipefail
OUT=/root/out-hbatch/c15
RP="python3 /root/out-hbatch/run_pool.py"
mkdir -p "$OUT"
PICKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,0"

date -u +%Y-%m-%dT%H:%M:%SZ > /root/TIMING-IN-FLIGHT
echo "hbatch-battery c=15 rung (owner: hbatch-battery agent)" >> /root/TIMING-IN-FLIGHT

for r in 1 2 3; do
  for arm in off on; do
    extras=()
    [ "$arm" = "on" ] && extras=(MEMRA_HYPER_BATCH=1)
    /root/out-hbatch/serve.sh start "c15-$r-$arm" "${extras[@]}" || { echo "C15_${r}_${arm}=BOOTFAIL"; continue; }
    $RP conc --n 15 --picks "$PICKS" --out "$OUT/l$r-$arm" || echo "WARN: c15 $r $arm errors"
  done
done
/root/out-hbatch/serve.sh stop
rm -f /root/TIMING-IN-FLIGHT
echo "=== C15 SUMMARY ==="
python3 - <<'PY'
import glob, json, statistics
for arm in ("off", "on"):
    rows = []
    for f in sorted(glob.glob(f"/root/out-hbatch/c15/l*-{arm}/conc-15-greedy.json")):
        j = json.load(open(f))
        rows.append(j)
        print(f"{arm} {f.split('/')[-2]}: agg={j['aggregate_tok_s']:.2f} dw={j['decode_window_tok_s']:.2f} "
              f"p50={j['per_session_tok_s_p50']:.1f} ttft_p50={j['ttft_p50_s']:.2f} ttft_max={j['ttft_max_s']:.2f} errs={j['rows_err']}")
    aggs = [r["aggregate_tok_s"] for r in rows]
    if aggs:
        med = statistics.median(aggs)
        print(f"{arm} c=15 MEDIAN agg={med:.2f} spread%={(max(aggs)-min(aggs))/med*100:.3f}")
PY
echo "=== LOOP-LAW ==="
python3 /root/out-hbatch/looplaw_screen.py "$OUT"/l*
echo "C15_DONE"
