#!/usr/bin/env bash
# run.sh <tag> <model> <rows> [VAR=VALUE ...]: one row-bisect run, detached by the caller.
set -uo pipefail
tag=$1; model=$2; rows=$3; shift 3
base=$HOME/.cache/memra-1203
out=$base/runs/$tag
mkdir -p "$out"
prompt=$base/tree/research/e2e/prompts/board-2048.txt
bin=$base/target/release/row-bisect
{
  echo "tag=$tag model=$(basename "$model") rows=$rows env=$*"
  echo "bin_sha256=$(sha256sum "$bin" | cut -d' ' -f1)"
  echo "prompt_sha256=$(sha256sum "$prompt" | cut -d' ' -f1)"
  echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
} > "$out/result.txt"
start=$(date +%s)
env "$@" "$bin" "$model" "$prompt" "$rows" --out "$out/taps" > "$out/log" 2>&1
rc=$?
echo "rc=$rc wall_s=$(( $(date +%s) - start ))" >> "$out/result.txt"
echo RUN-DONE >> "$out/result.txt"
