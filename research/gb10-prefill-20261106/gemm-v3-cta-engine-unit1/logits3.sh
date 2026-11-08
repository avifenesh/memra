#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: part 3 of gates-v3b.sh alone, run-gen pp3855 logits
# with v3 on, MEMRA_MMQ_F8F4_V3=0 and MEMRA_MMQ_F8F4_V2=0 (all three must be equal).
# usage: logits3.sh BIN OUT
set -u
V=$1; OUT=$2
D=$HOME/.cache/memra-f8f4v3
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
export TMPDIR=$D/tmp
P=$OUT/rollback; mkdir -p "$P"
python3 - "$P/prompt.txt" <<'EOF'
import sys
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
lines = [f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}." for i in range(146)]
open(sys.argv[1], "w").write("\n".join(lines))
EOF
for arm in v3 v3off v2off; do
  envs=()
  [ $arm = v3off ] && envs=(MEMRA_MMQ_F8F4_V3=0)
  [ $arm = v2off ] && envs=(MEMRA_MMQ_F8F4_V2=0)
  env "${envs[@]}" MEMRA_PROMPT_FILE=$P/prompt.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 MEMRA_PP_WARMUP=1 \
    MEMRA_PP_LOGITS=$P/$arm.logits "$V/run-gen" "$G8" > "$P/$arm.log" 2>&1
  echo "rollback $arm rc=$? $(grep -E 'pp-only MEDIAN' "$P/$arm.log") $(sha256sum "$P/$arm.logits" | cut -c1-16)" | tee -a "$OUT/logits.log"
done
echo LOGITS3-DONE | tee -a "$OUT/logits.log"
