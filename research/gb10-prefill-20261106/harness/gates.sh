#!/usr/bin/env bash
# Runs ON a GB10 host: the exactness battery for one banked binary dir on the Qwen3.8-27B GGUF.
# usage: gates.sh BIN OUT TREE
#   kernel-check x3 (device default, MEMRA_PRIME_KV_H2=0, MEMRA_PRIME_KV_T3=1), the calibrated
#   run-gen argmax-margin gate, run-spec K=1..8 self-consistency (T=0, 32 tokens).
set -u
BIN=$1; OUT=$2; TREE=$3
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p "$OUT"
GATE=$OUT/gate-root
rm -rf "$GATE"; mkdir -p "$GATE/target" "$GATE/research/e2e"
cp -a "$TREE/tools" "$GATE/tools"
cp -a "$TREE/research/e2e/prompts" "$GATE/research/e2e/prompts"
ln -s "$BIN" "$GATE/target/release"
cat "$BIN/COMMIT" > "$OUT/COMMIT"
KC=(--require-manifest "$TREE/tools/kernel-check-27b.cells" --require-manifest "$TREE/tools/kernel-check-step35.cells")
for arm in default h2off t3; do
  case $arm in
    default) envs=() ;;
    h2off) envs=(MEMRA_PRIME_KV_H2=0) ;;
    t3) envs=(MEMRA_PRIME_KV_T3=1) ;;
  esac
  (unset MEMRA_KC_FAST MEMRA_KC_ONLY; env "${envs[@]}" "$BIN/kernel-check" "$G8" "${KC[@]}") > "$OUT/kernel-check-$arm.log" 2>&1
  echo "kernel-check $arm rc=$? $(tail -1 "$OUT/kernel-check-$arm.log")" | tee -a "$OUT/gates.log"
  grep -E "fa_prefill_view_ws vs inline" "$OUT/kernel-check-$arm.log" | tee -a "$OUT/gates.log"
done
MEMRA_GATE_LOG_ROOT=$OUT bash "$GATE/tools/argmax-margin-gate.sh" "$G8" > "$OUT/argmax.log" 2>&1
echo "argmax rc=$? $(grep -E 'SUMMARY|PASS|FAIL' "$OUT/argmax.log" | tr '\n' ' ')" | tee -a "$OUT/gates.log"
(unset MEMRA_PROMPT_DIR MEMRA_SPEC_K MEMRA_GEN_ONLY; MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 "$BIN/run-spec" "$G8") > "$OUT/run-spec.log" 2>&1
echo "run-spec rc=$? $(grep -E 'SELF-CONSISTENCY|acceptance' "$OUT/run-spec.log" | tail -3 | tr '\n' ' ')" | tee -a "$OUT/gates.log"
echo GATES-DONE | tee -a "$OUT/gates.log"
