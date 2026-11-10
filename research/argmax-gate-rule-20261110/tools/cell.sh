#!/usr/bin/env bash
# cell.sh <tag> <probe> <model> <prompt> <window> <eager|batch> [<expected probe sha256>]
# One calibration cell: main's unmodified argmax-margin-probe over a wide window. Runs on the
# box that holds the GPU. Raw log first; the wall time and exit code go to result.txt.
set -uo pipefail
tag=$1; probe=$2; model=$3; prompt=$4; window=$5; arm=$6; want=${7:-}
out=$HOME/.cache/memra-gate-rule/cells/$tag
mkdir -p "$out"
have=$(sha256sum "$probe" | cut -d' ' -f1)
if [ -n "$want" ] && [ "$have" != "$want" ]; then
    echo "CELL-REFUSED $tag: probe sha256 $have, expected $want"; exit 3
fi
args=("$model" "$prompt" "$window")
[ "$arm" = batch ] && args+=(--decode batch)
{
    echo "tag=$tag window=$window arm=$arm"
    echo "probe_sha256=$(sha256sum "$probe" | cut -d' ' -f1)"
    echo "model=$(basename "$model") model_bytes=$(stat -L -c %s "$model")"
    echo "prompt_sha256=$(sha256sum "$prompt" | cut -d' ' -f1)"
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
} > "$out/result.txt"
start=$(date +%s)
"$probe" "${args[@]}" > "$out/probe.log" 2>&1
rc=$?
echo "probe_sha256_after=$(sha256sum "$probe" | cut -d' ' -f1)" >> "$out/result.txt"
echo "rc=$rc wall_s=$(( $(date +%s) - start ))" >> "$out/result.txt"
cat "$out/result.txt"
echo "CELL-DONE $tag rc=$rc"
