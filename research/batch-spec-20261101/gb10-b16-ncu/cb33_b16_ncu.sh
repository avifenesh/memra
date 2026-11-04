#!/usr/bin/env bash
# Runs ON the GB10 box: design check for the 16-row matmul (B16-MATMUL-DESIGN.md, "Measure
# before building"). ncu on the auto-picked singles at m=8 and m=16 over blk.0's NVFP4 tensors:
# kernel time, DRAM read bytes, L2 -> L1 read sectors, L1 hit rate, L2 and L1 throughput.
set -uo pipefail
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-b16ncu
mkdir -p "$OUT"
rm -f "$OUT/done.txt"
B=$HOME/.cache/memra-batchspec-target/release/mmvq-b16-bench
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
M="gpu__time_duration.sum,dram__bytes_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,l1tex__t_sector_hit_rate.pct,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active"
for m in 8 16; do
  sudo -E env PATH="$PATH" timeout 1200 "$(command -v ncu)" --metrics "$M" -k "regex:qmatvec_nvfp4_mmvq_b(8|16)_rpsc" --launch-count 12 --csv \
    "$B" "$G" --layers 1 --m $m --iters 1 > "$OUT/m$m.csv" 2> "$OUT/m$m.err"
  echo "m$m rc=$?" >> "$OUT/done.txt"
  sudo chown "$USER" "$OUT"/*
done
echo "STEP_DONE $OUT"
