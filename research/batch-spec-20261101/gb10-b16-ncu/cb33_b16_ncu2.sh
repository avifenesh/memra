#!/usr/bin/env bash
# Runs ON the GB10 box: stall attribution for the FFN singles at m=8 and m=16 (ffn_down, ffn_gate,
# ffn_up of blk.0: launches 2..4 of the rpsc kernel), boost clocks (no ncu clock control).
set -uo pipefail
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-b16ncu2
mkdir -p "$OUT"
B=$HOME/.cache/memra-batchspec-target/release/mmvq-b16-bench
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
NCU=$(command -v ncu)
SECS="--section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats --section InstructionStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section SourceCounters"
for m in 8 16; do
  sudo -E env PATH="$PATH" timeout 1200 "$NCU" --clock-control none -k "regex:qmatvec_nvfp4_mmvq_b(8|16)_rpsc" \
    --launch-skip 2 --launch-count 3 $SECS --import-source yes -o "$OUT/m$m" -f \
    "$B" "$G" --layers 1 --m $m --iters 1 > "$OUT/m$m.out" 2>&1
  echo "m$m rc=$?" >> "$OUT/done.txt"
  sudo "$NCU" --import "$OUT/m$m.ncu-rep" --page details --csv > "$OUT/m$m.csv" 2>/dev/null
  sudo "$NCU" --import "$OUT/m$m.ncu-rep" --page source --csv --print-source sass > "$OUT/m$m-sass.csv" 2>/dev/null
  sudo chown "$USER" "$OUT"/*
done
echo "STEP_DONE $OUT"
