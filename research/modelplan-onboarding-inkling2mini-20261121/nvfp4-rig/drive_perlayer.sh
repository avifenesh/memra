#!/bin/bash
# Lane M: per-layer NVFP4 attention arms (w:attn:L0..L23) on rawdoc and chatwrap in lane F's harness, in bounded
# leases (25 min of work each), then lane F's stats per arm into gpu-run/perlayer-*.json.
L=$(cd "$(dirname "$0")" && pwd)
K=$HOME/.cache/inkling2mini/klsens
O=$L/gpu-run
mkdir -p "$O"
cd "$K" || exit 1
for i in 1 2 3 4; do
  log=$O/perlayer-lease-$i.log
  gpulock run --perf --ttl 32m --wait 6h --for "inkling2 lane M: per-layer NVFP4 attention KL, rawdoc+chatwrap (lane F harness)" -- venv/bin/python klrun.py --jobs "$L/lanef-jobs-perlayer.json" --budget-s 1500 > "$log" 2>&1
  grep -q ALL_DONE "$log" && break
  grep -q BUDGET_STOP "$log" || { echo "perlayer lease $i failed, see $log"; exit 1; }
done
for set in rawdoc chatwrap; do
  for l in $(seq 0 23); do
    f=$K/results/arms/$set/${set}_lanem_w_attn_L$l
    [ -f "$f.npz" ] && venv/bin/python "$L/lanef_stats.py" "$f.npz" "$O/perlayer-$set-L$l.json" >> "$O/perlayer-stats.txt"
  done
done
echo PERLAYER_DONE >> "$O/perlayer-stats.txt"
