#!/usr/bin/env bash
# Runs ON a GB10 host: which prime attention kernel a binary launches, for run-gen's prime path
# (pp-only on a ~16k record prompt) and for one served cold prompt, from Nsight Systems kernel
# summaries. usage: which_kernel.sh BIN OUT [ENV...]
set -u
D=$HOME/.cache/memra-attn-gb10
BIN=$1; OUT=$2; shift 2
NSYS=/usr/local/cuda/bin/nsys
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p "$OUT"
python3 - "$OUT/prompt.txt" <<'EOF'
import sys
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
lines = [f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}." for i in range(600)]
open(sys.argv[1], "w").write("\n".join(lines))
EOF
env "$@" MEMRA_PP_PRIME=1 MEMRA_PROMPT_FILE=$OUT/prompt.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
  $NSYS profile -t cuda -s none --cpuctxsw=none --cuda-graph-trace=node --force-overwrite=true -o "$OUT/rungen" \
  "$BIN/run-gen" "$G8" > "$OUT/rungen.log" 2>&1
$NSYS stats --report cuda_gpu_kern_sum --format csv "$OUT/rungen.nsys-rep" > "$OUT/rungen-kern.csv" 2>&1
grep -E "fa_prefill|fa_dequant" "$OUT/rungen-kern.csv" | cut -c1-200
echo WHICH-DONE
