#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: build probe_gdn.cu against a tree's cu/hybrid.cu and run it.
# usage: run_probe_gdn.sh TREE OUTLOG [T ...]
set -u
TREE=$1; OUT=$2; shift 2
P=$(cd "$(dirname "$0")" && pwd)
cd "$P" || exit 1
nice -n 10 /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
  -I"$TREE/crates/memra-engine/cu" --resource-usage probe_gdn.cu -o probe_gdn > "$OUT.nvcc" 2>&1
echo "nvcc rc=$?" | tee "$OUT"
grep -E "k4_tpl|gdn_chunk_state_mma|gdn_chunk_output_mma" -A1 "$OUT.nvcc" | grep -E "Function properties|registers|spill" | head -n 40 >> "$OUT"
for T in "${@:-4096}"; do
  ./probe_gdn "$T" 7 >> "$OUT" 2>&1
done
echo RUN-PROBE-GDN-DONE >> "$OUT"
