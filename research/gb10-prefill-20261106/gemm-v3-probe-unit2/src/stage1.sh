#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's sparklock lease. Stage 1 of the F8F4 tile v3 design lane:
#   1. build the shipped v2 tile driver (round-3 mmq_probe.cu, PROBE_MODE 0 and 3) and time it per shape;
#   2. ncu --set full (sudo -n) on one gate+up launch in mode 0 and in mode 3, and on one down launch
#      in mode 0, plus raw CSV and SASS source pages;
#   3. the exact-epilogue ceiling probe (harness/ceil_probe3.cu) and the round-3 MMA probe, timed, and
#      ncu on the ceiling probe's 16-warp scalar kernel;
#   4. setmaxnreg at runtime (nreg_rt.cu); ptxas register and spill lines for every build.
# usage: stage1.sh DIR   (DIR holds this script, the sources and mmq_common.cuh)
set -u
export PATH="/usr/local/cuda/bin:$PATH"
P=$1; cd "$P" || exit 1
NV="nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr -Xptxas -v"
log(){ echo "$(date +%T) $*" | tee -a stage1.log; }
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader | tee -a stage1.log
nvcc --version | tail -n 1 | tee -a stage1.log
for m in 0 1 2 3; do
  $NV -DPROBE_MODE=$m driver_ncu.cu -o drv_m$m > build_m$m.log 2>&1 || log "BUILD-FAIL drv_m$m"
  grep -A2 "mul_mat_f8f4v2ILi256ELi128ELi8ELi2ELi64ELi3ELi1ELi0" build_m$m.log | grep -E "spill|Used" | sed "s/^/m$m /" | tee -a stage1.log
done
$NV ceil_probe3.cu -o ceil3 > build_ceil3.log 2>&1 || log "BUILD-FAIL ceil3"
$NV probe_mma_f8f4.cu -o pmma > build_pmma.log 2>&1 || log "BUILD-FAIL pmma"
$NV nreg_rt.cu -o nreg_rt > build_nreg.log 2>&1 || log "BUILD-FAIL nreg_rt"
log "timing (n=11 per shape and mode)"
for m in 0 1 2 3; do for s in 0 1 2 3; do ./drv_m$m $s 11 | head -n 1 | tee -a stage1.log; done; done
./ceil3 2>&1 | tee -a stage1.log
./pmma 2>&1 | tee -a stage1.log
./nreg_rt 2>&1 | tee -a stage1.log
NCU="sudo -n -E env PATH=$PATH ncu --set full --import-source no -f"
KRE='regex:mul_mat_f8f4v2'
log "ncu"
$NCU -k "$KRE" -s 1 -c 1 -o "$P/m0-gate" ./drv_m0 0 2 > ncu-m0-gate.log 2>&1; log "ncu m0 gate rc=$?"
$NCU -k "$KRE" -s 1 -c 1 -o "$P/m3-gate" ./drv_m3 0 2 > ncu-m3-gate.log 2>&1; log "ncu m3 gate rc=$?"
$NCU -k "$KRE" -s 1 -c 1 -o "$P/m0-down" ./drv_m0 1 2 > ncu-m0-down.log 2>&1; log "ncu m0 down rc=$?"
$NCU -k "regex:k_f8tile" -s 2 -c 1 -o "$P/ceil-f8tile" ./ceil3 > ncu-ceil.log 2>&1; log "ncu ceil rc=$?"
sudo -n chown "$USER" "$P"/*.ncu-rep 2>/dev/null
for r in m0-gate m3-gate m0-down ceil-f8tile; do
  [ -f "$r.ncu-rep" ] || continue
  ncu -i "$r.ncu-rep" --page details --print-units base --print-details all > "$r.txt" 2>&1
  ncu -i "$r.ncu-rep" --page raw --csv > "$r.raw.csv" 2>&1
  ncu -i "$r.ncu-rep" --page source --csv --print-source sass > "$r-sass.csv" 2>&1
done
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | tee -a stage1.log
log STAGE1-DONE
