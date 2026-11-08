#!/usr/bin/env bash
# Runs ON b23d under the caller's lease: the 16k argmax-margin gate on the follow-up binary with v3
# on (its default), the arm the cb33 lease ended before. Not a timing run.
# usage: argmax-fup.sh FOLLOWUP_BIN TREE OUT
set -u
NB=$1; TREE=$2; OUT=$3
D=$HOME/.cache/memra-f8f4v3
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
export TMPDIR=$D/tmp PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
R=$OUT/argmax16k-v3; mkdir -p "$R/root/target" "$R/root/research/e2e"
cp -a "$TREE/tools" "$R/root/tools"; cp -a "$TREE/research/e2e/prompts" "$R/root/research/e2e/prompts"
ln -s "$NB" "$R/root/target/release"
(cd "$R/root" && MEMRA_GATE_LOG_ROOT=$R bash tools/argmax-margin-gate.sh "$G8" --prompt research/e2e/prompts/p4-16k.txt --logdir "$R") > "$R/argmax.log" 2>&1
echo "$(date +%T) argmax-16k v3 rc=$? $(grep -E 'SUMMARY|PASS|FAIL|flip budget' "$R/argmax.log" | tr '\n' ' ')" | tee "$OUT/argmax.summary"
echo ARGMAX-FUP-DONE
