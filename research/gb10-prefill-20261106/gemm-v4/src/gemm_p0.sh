#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: lever 6 probe P0, the shipped F8F4 v3 tile measured where the
# design lane last measured v2. Builds the CTA-probe drivers (plain/residual and gate/up silu entries,
# image pass included) against the tree's mmq_nvfp4_w4a8.cu with -lineinfo, times v2 and v3 at T 4,096
# in both in-process orders, then ncu --set full on the v3 launches (one rep per shape) and exports the
# raw metrics and the SASS source page. usage: gemm_p0.sh OUTDIR
set -u
OUT=$1
D=$HOME/.cache/memra-gb10-r4
T=$D/tree
P=$D/gemm-p0
export PATH="/usr/local/cuda/bin:$PATH" TMPDIR=$D/tmp
mkdir -p "$OUT" "$P" "$TMPDIR"
cp "$T"/crates/memra-engine/cu/mmq_nvfp4_w4a8.cu "$T"/crates/memra-engine/cu/mmq_common.cuh "$P"/
cp "$T"/research/gb10-prefill-20261106/gemm-v3-cta-probe-unit1/plain-resid/drv_cta.cu "$P"/
cp "$T"/research/gb10-prefill-20261106/gemm-v3-cta-probe-unit1/gateup/drv_gateup.cu "$P"/
cp "$T"/research/gb10-prefill-20261106/gemm-v3-probe-unit2/src/ncu_summary.py "$T"/research/gb10-prefill-20261106/gemm-v3-probe-unit2/src/sass_stalls.py "$P"/
{ git -C "$T" log --oneline -1; (cd "$P" && sha256sum mmq_nvfp4_w4a8.cu mmq_common.cuh drv_cta.cu drv_gateup.cu); } > "$OUT/SOURCES"
for d in drv_cta drv_gateup; do
  (cd "$P" && nice -n 10 nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
     --resource-usage -o "$d" "$d.cu" -lcuda > "$OUT/build-$d.log" 2>&1; echo "build $d rc=$?" >> "$OUT/p0.log")
done
grep -A2 "Function properties for .*mul_mat_f8f4v3" "$OUT/build-drv_cta.log" "$OUT/build-drv_gateup.log" | grep -E "properties|Used|spill" >> "$OUT/p0.log"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader >> "$OUT/p0.log"
for ord in 01 10 01 10; do
  "$P/drv_cta" 4096 11 "$ord" >> "$OUT/time-cta.log" 2>&1
  "$P/drv_gateup" 4096 11 "$ord" >> "$OUT/time-gateup.log" 2>&1
done
echo "timing done" >> "$OUT/p0.log"
NCU="sudo -n -E env PATH=$PATH ncu --set full --import-source yes -f"
(cd "$P" && timeout 900 $NCU -k "regex:mul_mat_f8f4v3" -c 8 -o "$OUT/ncu-cta" ./drv_cta 4096 1 01 > "$OUT/ncu-cta.log" 2>&1; echo "ncu cta rc=$?" >> "$OUT/p0.log")
(cd "$P" && timeout 600 $NCU -k "regex:mul_mat_f8f4v3" -c 2 -o "$OUT/ncu-gateup" ./drv_gateup 4096 1 01 > "$OUT/ncu-gateup.log" 2>&1; echo "ncu gateup rc=$?" >> "$OUT/p0.log")
sudo -n chown "$USER" "$OUT"/*.ncu-rep 2>/dev/null
for r in ncu-cta ncu-gateup; do
  [ -f "$OUT/$r.ncu-rep" ] || continue
  ncu -i "$OUT/$r.ncu-rep" --page raw --csv > "$OUT/$r.raw.csv" 2>&1
  ncu -i "$OUT/$r.ncu-rep" --page source --csv --print-source sass > "$OUT/$r-sass.csv" 2>&1
done
echo "P0-DONE" >> "$OUT/p0.log"
