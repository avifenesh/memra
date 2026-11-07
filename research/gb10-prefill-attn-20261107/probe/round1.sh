#!/usr/bin/env bash
# Runs ON a GB10 host under this lane's lease: the attention ceiling probe, the candidate timing +
# byte compare at T = 4096 over four prefix depths, then ncu on h2 and the diagnostics.
set -u
D=$HOME/.cache/memra-attn-gb10
cd "$D/probe" || exit 1
mkdir -p "$D/logs"
{
  date -Is
  nice -n 10 /usr/local/cuda/bin/nvcc -O3 -gencode arch=compute_121a,code=sm_121a ceil_attn.cu -o ceil_attn && ./ceil_attn
  echo "ceil rc=$?"
} > "$D/logs/ceil-attn.log" 2>&1
bash "$D/probe/run_probe.sh" r1 4096 "4096,16384,65536,131072" 3 \
  "h2,h2c,sb2,ad,sbad,qh8,qh12,qh8db,fa2s,pp,nostage,nosm,qh8ns" > /dev/null 2>&1
for k in h2 nostage; do
  bash "$D/probe/ncu_probe.sh" "r1-$k" "$k" 4096 65536 > "$D/logs/ncu-r1-$k.log" 2>&1
done
echo ROUND1-DONE > "$D/logs/round1.done"
