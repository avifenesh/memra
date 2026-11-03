#!/usr/bin/env bash
# Runs ON the GB10 box: ncu sections for the decode kernels that stop scaling (b16 matvec, FA seqs).
set -uo pipefail
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
OUT=$HOME/memra-run/decode-ncu
mkdir -p "$OUT"
B=$HOME/.cache/memra-batchspec-target/release
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
NCU=$(command -v ncu)
SECS="--section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis"
run() { # tag seqs ctx kernel-regex [bench]
  sudo -E env PATH="$PATH" MEMRA_DECODE_BATCH_CAP=16 MEMRA_DBB_SAMP=greedy \
    "$NCU" --profile-from-start off -k "regex:$4" --launch-skip 8 --launch-count 6 $SECS \
    -o "$OUT/$1" -f "${5:-$B/decode-batch-bench}" "$G" --seqs $2 --chunk $(( $2 > 16 ? 16 : $2 )) --ctx $3 --steps 2 --reps 1 \
    > "$OUT/$1.out" 2>&1
  echo "$1 rc=$?" >> "$OUT/runs.txt"
  sudo "$NCU" --import "$OUT/$1.ncu-rep" --page details --csv > "$OUT/$1.csv" 2>/dev/null
  sudo chown "$USER" "$OUT"/*
}
run b16-mm-old 16 2048 '^qmatvec_nvfp4_mmvq_b16_rp$' "$HOME/memra-run/dbb-old"
run b16-mm 16 2048 '^qmatvec_nvfp4_mmvq_b16_rpsc$'
run b16-g4 16 2048 '^qmatvec_nvfp4_mmvq_group4_b16_rp$'
run b8-mm 8 2048 '^qmatvec_nvfp4_mmvq_b8_rpsc$'
run b8-fa 8 12288 '^fa_decode_vec_q_seqs_v4$'
echo "NCU_DONE $OUT"
