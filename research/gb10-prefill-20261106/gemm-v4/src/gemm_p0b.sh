#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: lever 6 probe P0b, the fused gate/up silu launch (the largest
# GEMM bucket) on v2 and v3: timing at T 4,096 in both orders, then ncu --set full on its v3 launch.
# usage: gemm_p0b.sh OUTDIR
set -u
OUT=$1
D=$HOME/.cache/memra-gb10-r4
P=$D/gemm-p0
export PATH="/usr/local/cuda/bin:$PATH" TMPDIR=$D/tmp
mkdir -p "$OUT"
cp "$D/drv_p0.cu" "$P/"
(cd "$P" && sha256sum mmq_nvfp4_w4a8.cu mmq_common.cuh drv_p0.cu) > "$OUT/SOURCES"
(cd "$P" && nice -n 10 nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
   --resource-usage -o drv_p0 drv_p0.cu -lcuda > "$OUT/build-drv_p0.log" 2>&1; echo "build drv_p0 rc=$?" >> "$OUT/p0.log")
grep -A2 "Function properties for .*mul_mat_f8f4v3.*Li1EE" "$OUT/build-drv_p0.log" | grep -E "properties|Used|spill" >> "$OUT/p0.log"
for ord in 01 10 01 10; do "$P/drv_p0" 4096 11 "$ord" 4 >> "$OUT/time-gateup.log" 2>&1; done
echo "timing done" >> "$OUT/p0.log"
NCU="sudo -n -E env PATH=$PATH ncu --set full --import-source yes -f"
(cd "$P" && timeout 600 $NCU -k "regex:mul_mat_f8f4v3" -c 2 -o "$OUT/ncu-gateup" ./drv_p0 4096 1 01 4 > "$OUT/ncu-gateup.log" 2>&1; echo "ncu gateup rc=$?" >> "$OUT/p0.log")
sudo -n chown "$USER" "$OUT"/*.ncu-rep 2>/dev/null
if [ -f "$OUT/ncu-gateup.ncu-rep" ]; then
  ncu -i "$OUT/ncu-gateup.ncu-rep" --page raw --csv > "$OUT/ncu-gateup.raw.csv" 2>&1
  ncu -i "$OUT/ncu-gateup.ncu-rep" --page source --csv --print-source sass > "$OUT/ncu-gateup-sass.csv" 2>&1
fi
echo "P0b-DONE" >> "$OUT/p0.log"
