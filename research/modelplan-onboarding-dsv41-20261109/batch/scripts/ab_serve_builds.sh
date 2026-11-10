#!/bin/bash
# Two builds of memra-server on one card, A B B A, one lease a run: the 9-layer resident trunk, batching on, the
# shared client's decode_c1 and conc8, 3 samples. lease.sh records the kernel log's last page-cache drops before and
# after every run.
# usage: ab_serve_builds.sh TAG_A TAG_B CARD
set -u
A=$1; B=$2; CARD=$3
S=$LANE/spec
i=0
for arm in $A $B $B $A; do
  i=$((i + 1))
  TBIN=$S/bin/$arm EXTRA_BENCH=--no-smoke-gate bash $S/scripts/lease.sh abs-r$i 900 $CARD bash $S/scripts/serve_bench.sh \
    abs-$A-$B/r$i-$arm decode_c1,conc8 3 MEMRA_DSV41_TRUNK_LAYERS=9 MEMRA_CTX=8192 > $S/logs/abs-$A-$B-r$i-$arm.log 2>&1
  echo "$(date +%H:%M:%S) run $i $arm rc $? $(grep -h "page-cache drops" $S/logs/abs-$A-$B-r$i-$arm.log | tail -n 1 | cut -c10-200)"
done
python3 $S/scripts/ab_summary.py $S/out/abs-$A-$B
python3 - $S/out/abs-$A-$B <<'PY'
import glob, re, statistics as st, sys
for d in sorted(glob.glob(sys.argv[1] + "/r*")):
    by = {}
    for l in open(d + "/server.log", errors="ignore"):
        m = re.search(r"calls=(\d+) prefill=(\d+) decode_rows=(\d+) step_s=([0-9.]+)", l)
        if m and m.group(2) == "0":
            by.setdefault(int(m.group(3)), []).append(float(m.group(4)))
    print(d.rsplit("/", 1)[-1], "served step ms:", {k: round(st.mean(v) * 1e3, 2) for k, v in sorted(by.items()) if len(v) > 50})
PY
echo "AB-SERVE-DONE $A $B"
