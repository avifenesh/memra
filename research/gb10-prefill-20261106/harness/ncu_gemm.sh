#!/usr/bin/env bash
# Runs ON the GB10 host: ncu on the F8F4 prefill GEMM (and attention) in a monolithic run-gen
# prefill of a ~4096-token prompt. usage: ncu_gemm.sh OUT BIN [kernel-regex] [skip] [count]
set -u
OUT=$1; BIN=$2; KRE=${3:-mul_mat_q_nvfp4_f8f4}; SKIP=${4:-0}; COUNT=${5:-6}
D=$HOME/.cache/memra-prefill
mkdir -p "$OUT"
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
python3 - "$OUT/prompt.txt" <<'EOF'
import sys
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
lines = [f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}." for i in range(146)]
open(sys.argv[1], "w").write("\n".join(lines))
EOF
export MEMRA_PROMPT_FILE=$OUT/prompt.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0
sudo -E env PATH="$PATH" /usr/local/cuda/bin/ncu --target-processes all -k "regex:$KRE" -s "$SKIP" -c "$COUNT" \
  --set full -f -o "$OUT/ncu" "$BIN/run-gen" "$G8" > "$OUT/ncu.log" 2>&1
echo "rc=$?" >> "$OUT/ncu.log"
sudo chown "$USER" "$OUT"/ncu.ncu-rep 2>/dev/null
/usr/local/cuda/bin/ncu -i "$OUT/ncu.ncu-rep" --page details --print-units base > "$OUT/ncu-details.txt" 2>&1
echo done >> "$OUT/ncu.log"
