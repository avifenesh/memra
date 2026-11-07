#!/usr/bin/env bash
# Runs ON a GB10 unit under the caller's lease: decode tick of the bf16 GDN state storage lane,
# same binary, MEMRA_GDN_STATE_BF16=0/1, A B B A A B per cell, Qwen3.8-27B GGUF, greedy,
# 16 steps x 3 reps; token fingerprints per run (fp32 runs must match each other, bf16 runs
# each other).
#   gb10_bench.sh BINTAG
set -u
R=$HOME/.cache/memra-gdnbf16-real
B=$R/bins/${1:?bintag}
OUT=$R/runs/bench
mkdir -p "$OUT"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
run() { # tag bf16 seqs depth
  local f=$OUT/dbb-$1.log
  MEMRA_GDN_STATE_BF16=$2 MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 "$B/decode-batch-bench" "$G" \
    --seqs $3 --chunk $3 --ctx $4 --steps 16 --reps 3 > "$f" 2>&1
  echo "rc=$?" >> "$f"
  echo "$1 bf16=$2 s=$3 d=$4: $(grep CHUNKSWEEP "$f" | cut -c1-220) fp=$(grep TOKFP "$f" | sed 's/.*\[//' | md5sum | cut -c1-12) $(grep -m1 -iE 'out of memory|panicked' "$f" | cut -c1-120) | MemAvailable $(awk '/MemAvailable/{printf "%.1fGiB", $2/1048576}' /proc/meminfo)" >> "$OUT/ab.txt"
}
for cell in 1:2048 8:2048 16:2048 32:2048 16:12288; do
  s=${cell%%:*}; d=${cell#*:}
  i=0
  for arm in 0 1 1 0 0 1; do
    i=$((i + 1)); run "s$s-d$d-r$i" $arm $s $d
  done
done
echo "BENCH_DONE $OUT"
