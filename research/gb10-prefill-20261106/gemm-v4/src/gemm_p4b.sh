#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: lever 6 probe P4, the half-height last wave (mmq_v3t.cu, made by
# make_v3t.py from the tree's TU) against the shipped v3 through the engine entries, T 4,096, all five
# shapes, both in-process orders, twice; outputs compared byte for byte. usage: gemm_p1.sh OUTDIR
set -u
OUT=$1
D=$HOME/.cache/memra-gb10-r4
T=$D/tree
P=$D/gemm-p4b
export PATH="/usr/local/cuda/bin:$PATH" TMPDIR=$D/tmp
mkdir -p "$OUT" "$P"
cp "$T"/crates/memra-engine/cu/mmq_common.cuh "$P"/
cp "$D/make_v3t.py" "$D/drv_p4b.cu" "$P"/
python3 "$P/make_v3t.py" "$T/crates/memra-engine/cu/mmq_nvfp4_w4a8.cu" "$P/mmq_v3t.cu" >> "$OUT/done.log" 2>&1
{ git -C "$T" log --oneline -1; (cd "$P" && sha256sum mmq_v3t.cu mmq_common.cuh drv_p4b.cu make_v3t.py); } > "$OUT/SOURCES"
(cd "$P" && nice -n 10 nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
   --resource-usage -o drv_p4b drv_p4b.cu -lcuda > "$OUT/build.log" 2>&1; echo "build rc=$?" >> "$OUT/done.log")
grep -A2 "Function properties for .*mul_mat_f8f4v3" "$OUT/build.log" | grep -E "properties|Used|spill" >> "$OUT/done.log"
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.draw --format=csv,noheader >> "$OUT/done.log"
for ord in 01 10 01 10; do "$P/drv_p4b" 4096 41 "$ord" >> "$OUT/time.log" 2>&1; done
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.draw --format=csv,noheader >> "$OUT/done.log"
echo "P4b-DONE" >> "$OUT/done.log"
