#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: sizing the h2p attention lever. The shipped h2p, its probe
# clone (m3off) and the mode-3 twin (m3: the three ring stages load once, no waits or refills after;
# timing only) at T 4,096 over depths 32k-256k, then ncu --set full on h2p and m3 at depth 131,072.
# usage: attn_m3.sh OUTDIR
set -u
OUT=$1
D=$HOME/.cache/memra-gb10-r4
T=$D/tree
P=$D/attn6
export PATH="/usr/local/cuda/bin:$PATH" TMPDIR=$D/tmp
mkdir -p "$OUT" "$P"
cp "$T/crates/memra-engine/cu/flash_attn.cu" "$P/"
cp "$D/attn_pp.cuh" "$D/attn_ex2.cuh" "$D/attn_m3.cuh" "$D/probe_m3.cu" "$P/"
{ git -C "$T" log --oneline -1; (cd "$P" && sha256sum flash_attn.cu attn_pp.cuh attn_ex2.cuh attn_m3.cuh probe_m3.cu); } > "$OUT/SOURCES"
(cd "$P" && nice -n 10 nvcc -O3 -gencode arch=compute_121a,code=sm_121a -I. -I"$T/crates/memra-engine/cu" -lineinfo \
   --resource-usage probe_m3.cu -o probe_m3 > "$OUT/build.log" 2>&1; echo "build rc=$?" >> "$OUT/done.log")
grep -A2 -E "Function properties for (fa_prefill_qw_h2p|cand_m3)" "$OUT/build.log" | grep -E "properties|Used|spill" >> "$OUT/done.log"
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.draw --format=csv,noheader >> "$OUT/done.log"
for pass in 1 2; do
  (cd "$P" && timeout 900 ./probe_m3 4096 32768,65536,131072,262144 3 h2p,m3off,m3) >> "$OUT/time.log" 2>&1
  (cd "$P" && timeout 900 ./probe_m3 4096 32768,65536,131072,262144 3 h2p,m3,m3off) >> "$OUT/time.log" 2>&1
done
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.draw --format=csv,noheader >> "$OUT/done.log"
NCU="sudo -n -E env PATH=$PATH ncu --set full --import-source yes -f"
(cd "$P" && timeout 900 $NCU -k "regex:^(fa_prefill_qw_h2p|cand_m3)$" -c 4 -o "$OUT/ncu-attn" ./probe_m3 4096 131072 1 h2p,m3 > "$OUT/ncu.log" 2>&1; echo "ncu rc=$?" >> "$OUT/done.log")
sudo -n chown "$USER" "$OUT"/*.ncu-rep 2>/dev/null
if [ -f "$OUT/ncu-attn.ncu-rep" ]; then
  ncu -i "$OUT/ncu-attn.ncu-rep" --page raw --csv > "$OUT/ncu-attn.raw.csv" 2>&1
  ncu -i "$OUT/ncu-attn.ncu-rep" --page source --csv --print-source sass > "$OUT/ncu-attn-sass.csv" 2>&1
fi
echo "ATTN-DONE" >> "$OUT/done.log"
