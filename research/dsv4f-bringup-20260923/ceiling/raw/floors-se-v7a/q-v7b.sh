#!/usr/bin/env bash
# q-v7b.sh (SE pair): calibrate nsys's DRAM Read Bandwidth percentage on this card with the
# streaming probe (bw_read.cu, known bytes and time), so q-v7a's replayed-step percentage converts
# to bytes per step.
set -u
S=/root/rcpt/q-v7b.summary; R=/root/rcpt/gm-calib-v7b; mkdir -p $R
until grep -q V7A_DONE /root/rcpt/q-v7a.summary 2>/dev/null; do sleep 60; done
export PATH=/usr/local/cuda/bin:$PATH
cp /root/box/bw_read.cu $R/ && nvcc -O3 -arch=sm_120a -o $R/bw_read $R/bw_read.cu > $R/build.log 2>&1
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
CUDA_VISIBLE_DEVICES=0 $R/bw_read > $R/plain.txt 2>&1
CUDA_VISIBLE_DEVICES=0 nsys profile --gpu-metrics-devices=cuda-visible --gpu-metrics-set=gb20x --gpu-metrics-frequency=20000 -t cuda -f true -o $R/gm $R/bw_read > $R/nsys.txt 2>&1
nsys export --type sqlite -f true -o $R/gm.sqlite $R/gm.nsys-rep > /dev/null 2>&1
python3 - $R/gm.sqlite > $R/calib.txt 2>&1 <<'PY'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
names = {(t, m): n for t, m, n in c.execute("select typeId, metricId, metricName from TARGET_INFO_GPU_METRICS")}
rd = [k for k, n in names.items() if n.startswith("DRAM Read Bandwidth")][0]
for name, lo, hi in c.execute("select s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName order by k.start"):
    dur = (hi - lo) / 1e9
    vals = [v for (v,) in c.execute("select value from GPU_METRICS where typeId=? and metricId=? and timestamp between ? and ?", (rd[0], rd[1], lo, hi))]
    if len(vals) >= 3:
        print(f"KERNEL {name} dur_ms={dur*1e3:.3f} dram_read_pct_mean={sum(vals)/len(vals):.2f} samples={len(vals)}")
PY
rm -f $R/gm.sqlite
echo "calib $(grep -h READ $R/plain.txt | tr '\n' ' ') | $(sort -t= -k2 -n $R/calib.txt | tail -4 | tr '\n' ' ')" >> $S
echo V7B_DONE >> $S
