#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: the issue-after-compute variants against issue-end,
# two passes in opposite orders (each variant in its own process, shipped tile timed beside it,
# outputs compared bit for bit, 5 shapes at T 4,096, 11 launches), then ncu --set full on one gate
# launch of each. usage: stage5.sh DIR
set -u
export PATH="/usr/local/cuda/bin:$PATH"
P=$1; cd "$P" || exit 1
log(){ echo "$(date +%T) $*" | tee -a stage5.log; }
: > stage5.log
nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr -Xptxas -v tma_probe.cu -lcuda -o tma_probe > build5.log 2>&1 || { log BUILD-FAIL; exit 1; }
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | tee -a stage5.log
for v in 1 7 8 9 10 10 9 8 7 1; do
  log "variant bit $v"
  timeout 180 ./tma_probe 4096 11 -1 $((1 << v)) 2>&1 | grep -v "shipped (again)" | tee -a stage5.log
done
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | tee -a stage5.log
NCU="sudo -n -E env PATH=$PATH ncu --set full --import-source no -f"
for v in 1 7 8 9 10; do
  timeout 300 $NCU -k "regex:mul_mat_f8f4v3" -s 1 -c 1 -o "$P/s5-b$v-gate" ./tma_probe 4096 2 0 $((1 << v)) > ncu-s5-b$v.log 2>&1
  log "ncu bit $v rc=$?"
done
sudo -n chown "$USER" "$P"/*.ncu-rep 2>/dev/null
for v in 1 7 8 9 10; do
  [ -f "s5-b$v-gate.ncu-rep" ] || continue
  ncu -i "s5-b$v-gate.ncu-rep" --page raw --csv > "s5-b$v-gate.raw.csv" 2>&1
  ncu -i "s5-b$v-gate.ncu-rep" --page source --csv --print-source sass > "s5-b$v-gate-sass.csv" 2>&1
done
log STAGE3-DONE
