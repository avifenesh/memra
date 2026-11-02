#!/usr/bin/env bash
# Runs ON spark-cb33 under the caller's lease: GB10 correctness cells for one 121a build.
#   BIN=~/memra-run/bin-<sha12> correctness.sh      Log: stdout
set -u
BIN=${BIN:?bin}
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
GG=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
NV=$HOME/models/Qwen3.8-27B-NVFP4
PR=$HOME/projects/memra-prompts
cd "$HOME/projects/memra-spark" || exit 2   # kernel-check reads tools/*.cells manifests from the tree
sha256sum "$BIN"/memra-server "$BIN"/run-gen "$BIN"/run-spec "$BIN"/kernel-check
echo "##### kernel-check"
timeout 1800 "$BIN/kernel-check" 2>&1 | grep -E 'ALL GREEN|FAIL|SKIP|cells' | tail -25
echo "##### kernel-check rc=${PIPESTATUS[0]}"
for M in "$GG" "$NV"; do
  for p in "$PR"/p*.txt; do
    echo "##### run-gen $(basename "$M") $(basename "$p")"
    MEMRA_CHAT=1 MEMRA_NGEN=${NGEN:-64} MEMRA_PROMPT_FILE="$p" timeout 1200 "$BIN/run-gen" "$M" 2>&1 \
      | grep -E 'MATCH|MISMATCH|prefill .* tok/s|generated|Error' | tail -4
    echo "##### rc=${PIPESTATUS[0]}"
  done
done
echo "##### run-spec K=1..8 (GGUF trunk + embedded MTP)"
MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_CHAT=1 MEMRA_PROMPT_FILE="$PR/p1.txt" timeout 1500 "$BIN/run-spec" "$GG" 2>&1 \
  | grep -E 'K=|PASS|FAIL|self-consist|Error' | tail -14
echo "##### run-spec rc=${PIPESTATUS[0]}"
