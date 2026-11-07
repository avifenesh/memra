#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's sparklock lease: build the v3 TMA probe, then per variant
# (own process, 180 s cap) time it against the shipped v2 tile with a bit-for-bit output compare
# (5 shapes, T 4,096, 11 launches each); then T 1,024 and 130 on the gate shape; then the co-issue
# probe; then ncu --set full on one launch of each v3 variant that ran clean. usage: stage2.sh DIR
set -u
export PATH="/usr/local/cuda/bin:$PATH"
P=$1; cd "$P" || exit 1
log(){ echo "$(date +%T) $*" | tee -a stage2.log; }
: > stage2.log
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | tee -a stage2.log
nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr -Xptxas -v tma_probe.cu -lcuda -o tma_probe > build.log 2>&1 || { log BUILD-FAIL; exit 1; }
grep -A3 "Compiling entry function '_Z14mul_mat_f8f4v" build.log | grep -E "Compiling|spill|Used" | sed -E 's/.*(f8f4v[23]I[^ ]*).*/\1/' | tee -a stage2.log
ok=""
for v in 0 1 2 3 4; do
  log "variant $v, T 4096, all shapes"
  timeout 180 ./tma_probe 4096 11 -1 $((1 << v)) 2>&1 | tee -a stage2.log
  rc=${PIPESTATUS[0]}; log "variant $v rc=$rc"
  if [ "$rc" = 0 ]; then ok="$ok $v"; fi
done
for v in $ok; do
  for T in 1024 130; do timeout 120 ./tma_probe $T 11 0 $((1 << v)) 2>&1 | tee -a stage2.log; done
done
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | tee -a stage2.log
NCU="sudo -n -E env PATH=$PATH ncu --set full --import-source no -f"
for v in $ok; do
  timeout 300 $NCU -k "regex:mul_mat_f8f4v3" -s 1 -c 1 -o "$P/v3p$v-gate" ./tma_probe 4096 2 0 $((1 << v)) > ncu-v3p$v.log 2>&1
  log "ncu v3 prod $v rc=$?"
done
sudo -n chown "$USER" "$P"/*.ncu-rep 2>/dev/null
for v in $ok; do
  [ -f "v3p$v-gate.ncu-rep" ] || continue
  ncu -i "v3p$v-gate.ncu-rep" --page raw --csv > "v3p$v-gate.raw.csv" 2>&1
  ncu -i "v3p$v-gate.ncu-rep" --page source --csv --print-source sass > "v3p$v-gate-sass.csv" 2>&1
done
log STAGE2-DONE
