#!/usr/bin/env bash
# Runs ON the GB10 host: interleaved run-gen prefill-only A/B for prefill-GEMM kernel levers.
# usage: gemm_ab.sh OUTDIR LINES name=bindir ...   (cells run in the order given)
# Each cell: one process, 1 warmup + 5 timed monolithic forwards of the record prompt
# (LINES lines, ~28 tokens each), last-row logits dumped (byte-compare across arms).
set -u
OUT=$1; LINES=$2; shift 2
mkdir -p "$OUT"
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
python3 - "$OUT/prompt.txt" "$LINES" <<'EOF'
import sys
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
n = int(sys.argv[2])
lines = [f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}." for i in range(n)]
open(sys.argv[1], "w").write("\n".join(lines))
EOF
nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw --format=csv,noheader -lms 250 > "$OUT/gpu.csv" 2>&1 &
smi=$!
i=0
for spec in "$@"; do
  i=$((i+1)); name=${spec%%=*}; bin=${spec#*=}
  log=$OUT/$(printf '%02d' $i)-$name.log
  echo "$(date -Is) cell $i $name $bin" | tee -a "$OUT/ab.log"
  MEMRA_PROMPT_FILE=$OUT/prompt.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 \
    MEMRA_PP_LOGITS=$OUT/$(printf '%02d' $i)-$name.logits "$bin/run-gen" "$G8" > "$log" 2>&1
  echo "rc=$?" >> "$log"
  grep -E "pp-only (rep|MEDIAN)" "$log" | sed "s/^/$name /" | tee -a "$OUT/ab.log"
  sha256sum "$OUT/$(printf '%02d' $i)-$name.logits" | sed "s/^/$name /" | tee -a "$OUT/ab.log"
done
kill $smi 2>/dev/null
echo "$(date -Is) done" | tee -a "$OUT/ab.log"
