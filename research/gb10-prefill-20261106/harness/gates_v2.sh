#!/usr/bin/env bash
# Runs ON a GB10 host: the exactness battery for the F8F4 v2 binary (gates.sh) plus the cells that
# reach the v2 route, whose threshold (2,048 rows) the default argmax prompt does not clear:
#   - argmax-margin gate on the 16k prompt (p4-16k) for v2 and for the base binary, and a byte
#     compare of the two probe tables (same prefill program => same table);
#   - the MEMRA_MMQ_F8F4_V2=0 rollback: run-gen pp3855 logits must equal the default's.
# usage: gates_v2.sh V_BIN BASE_BIN OUT TREE
set -u
V=$1; B=$2; OUT=$3; TREE=$4
D=$HOME/.cache/memra-gemm-v2
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p "$OUT"
bash "$D/gates.sh" "$V" "$OUT" "$TREE"
for arm in V B; do
  bin=$V; [ $arm = B ] && bin=$B
  R=$OUT/argmax16k-$arm; rm -rf "$R"; mkdir -p "$R/root/target" "$R/root/research/e2e"
  cp -a "$TREE/tools" "$R/root/tools"; cp -a "$TREE/research/e2e/prompts" "$R/root/research/e2e/prompts"
  ln -s "$bin" "$R/root/target/release"
  (cd "$R/root" && MEMRA_GATE_LOG_ROOT=$R bash tools/argmax-margin-gate.sh "$G8" --prompt research/e2e/prompts/p4-16k.txt --logdir "$R") > "$R/argmax.log" 2>&1
  echo "argmax-16k $arm rc=$? $(grep -E 'SUMMARY|PASS|FAIL' "$R/argmax.log" | tr '\n' ' ')" | tee -a "$OUT/gates.log"
done
ta=$(ls "$OUT"/argmax16k-V/gate-*.log 2>/dev/null | head -n 1); tb=$(ls "$OUT"/argmax16k-B/gate-*.log 2>/dev/null | head -n 1)
if [ -n "$ta" ] && [ -n "$tb" ]; then
  grep -vE 'ms|s/tok|tok/s|elapsed|wall|time' "$ta" > "$OUT/argmax16k-V.table"
  grep -vE 'ms|s/tok|tok/s|elapsed|wall|time' "$tb" > "$OUT/argmax16k-B.table"
  if cmp -s "$OUT/argmax16k-V.table" "$OUT/argmax16k-B.table"; then r=IDENTICAL; else r=DIFFER; fi
  echo "argmax-16k probe table V vs B: $r ($(sha256sum < "$OUT/argmax16k-V.table" | cut -c1-16) / $(sha256sum < "$OUT/argmax16k-B.table" | cut -c1-16))" | tee -a "$OUT/gates.log"
fi
P=$OUT/rollback; mkdir -p "$P"
python3 - "$P/prompt.txt" <<'EOF'
import sys
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
lines = [f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}." for i in range(146)]
open(sys.argv[1], "w").write("\n".join(lines))
EOF
for arm in on off; do
  env_v2=(); [ $arm = off ] && env_v2=(MEMRA_MMQ_F8F4_V2=0)
  env "${env_v2[@]}" MEMRA_PROMPT_FILE=$P/prompt.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 MEMRA_PP_WARMUP=1 \
    MEMRA_PP_LOGITS=$P/$arm.logits "$V/run-gen" "$G8" > "$P/$arm.log" 2>&1
  echo "rollback $arm rc=$? $(grep -E 'pp-only MEDIAN' "$P/$arm.log") $(sha256sum "$P/$arm.logits" | cut -c1-16)" | tee -a "$OUT/gates.log"
done
echo GATES-V2-DONE | tee -a "$OUT/gates.log"
