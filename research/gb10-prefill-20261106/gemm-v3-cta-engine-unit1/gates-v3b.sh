#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: parts 2-3 of gates-v3.sh (part 1, kernel-check x3,
# argmax and run-spec, ran on unit 2 before the unit went to a landing).
#   2. the argmax-margin gate on the 16k prompt (it primes through the v2/v3 route), with v3 on and with
#      MEMRA_MMQ_F8F4_V3=0, and a byte compare of the two probe tables;
#   3. run-gen pp3855 logits with v3 on, MEMRA_MMQ_F8F4_V3=0 and MEMRA_MMQ_F8F4_V2=0: all three equal.
# usage: gates-v3b.sh BIN OUT TREE
set -u
V=$1; OUT=$2; TREE=$3
D=$HOME/.cache/memra-f8f4v3
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
export TMPDIR=$D/tmp
mkdir -p "$OUT"
for arm in v3 v2; do
  envs=(); [ $arm = v2 ] && envs=(MEMRA_MMQ_F8F4_V3=0)
  R=$OUT/argmax16k-$arm; rm -rf "$R"; mkdir -p "$R/root/target" "$R/root/research/e2e"
  cp -a "$TREE/tools" "$R/root/tools"; cp -a "$TREE/research/e2e/prompts" "$R/root/research/e2e/prompts"
  ln -s "$V" "$R/root/target/release"
  (cd "$R/root" && env "${envs[@]}" MEMRA_GATE_LOG_ROOT=$R bash tools/argmax-margin-gate.sh "$G8" --prompt research/e2e/prompts/p4-16k.txt --logdir "$R") > "$R/argmax.log" 2>&1
  echo "argmax-16k $arm rc=$? $(grep -E 'SUMMARY|PASS|FAIL|flip budget' "$R/argmax.log" | tr '\n' ' ')" | tee -a "$OUT/gates.log"
done
ta=$(ls "$OUT"/argmax16k-v3/gate-*.log 2>/dev/null | head -n 1); tb=$(ls "$OUT"/argmax16k-v2/gate-*.log 2>/dev/null | head -n 1)
if [ -n "$ta" ] && [ -n "$tb" ]; then
  grep -vE 'ms|s/tok|tok/s|elapsed|wall|time' "$ta" > "$OUT/argmax16k-v3.table"
  grep -vE 'ms|s/tok|tok/s|elapsed|wall|time' "$tb" > "$OUT/argmax16k-v2.table"
  if cmp -s "$OUT/argmax16k-v3.table" "$OUT/argmax16k-v2.table"; then r=IDENTICAL; else r=DIFFER; fi
  echo "argmax-16k probe table v3 vs v2: $r ($(sha256sum < "$OUT/argmax16k-v3.table" | cut -c1-16) / $(sha256sum < "$OUT/argmax16k-v2.table" | cut -c1-16))" | tee -a "$OUT/gates.log"
fi
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
  echo "rollback $arm rc=$? $(grep -E 'pp-only MEDIAN' "$P/$arm.log") $(sha256sum "$P/$arm.logits" | cut -c1-16)" | tee -a "$OUT/gates.log"
done
echo GATES-V3-DONE | tee -a "$OUT/gates.log"
