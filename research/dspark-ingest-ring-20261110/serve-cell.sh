#!/usr/bin/env bash
# One served cell (memra #1159): wait for the card to read under 75 C, then the repo's
# tools/dspark-serve-smoke.sh. Runs inside the caller's GPU lease.
# usage: serve-cell.sh <memra_checkout> <memra-server> <model.gguf> <draft_dir> <evidence_dir> [VAR=VALUE ...]
set -u
repo=$1 bin=$2 model=$3 draft=$4 ev=$5
shift 5
waitfor --quiet --until '[ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -lt 75 ]' --every 10s --timeout 6m
echo "card $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C at start"
echo "server sha256 $(sha256sum "$bin" | cut -d' ' -f1)"
echo "+ env $* dspark-serve-smoke.sh $(basename "$model") $(basename "$draft") memra-server"
env "$@" bash "$repo/tools/dspark-serve-smoke.sh" "$model" "$draft" "$bin" "$ev"
