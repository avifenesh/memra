#!/usr/bin/env bash
# One dspark_q38_gate cell (memra #1159): wait for the card to read under 75 C, then run the
# gate. Runs inside the caller's GPU lease.
# usage: cell.sh <dspark_q38_gate> <model.gguf> <draft_dir> [VAR=VALUE ...]
set -u
bin=$1 model=$2 draft=$3
shift 3
waitfor --quiet --until '[ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -lt 75 ]' --every 10s --timeout 6m
echo "card $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C at start"
echo "binary sha256 $(sha256sum "$bin" | cut -d' ' -f1)"
echo "+ env $* dspark_q38_gate $(basename "$model") $(basename "$draft")"
env "$@" "$bin" "$model" "$draft"
