#!/usr/bin/env bash
# red_green.sh <built argmax-margin-probe> <gate log> <out dir>
# The stored-rows check on a real 12-row gate table of the RTX 5090 Laptop GPU's Ornith eager
# cell, measured by another build than the stored rows (v0.141.0's probe; the stored rows are
# main 031f146690's calibration). Three tables through the probe's --judge with the floors table
# and tools/argmax-margin-rows/:
#   green  the table as measured
#   red-1  one margin moved by 1 logit (decode margin of the last row)
#   red-2  a token change at a near tie (row 2037: decode picks another id at margin 0.05)
# Each verdict goes to <out dir>/<arm>.txt with its exit code.
set -uo pipefail
probe=$(realpath "$1"); log=$(realpath "$2"); out=$(realpath -m "$3")
cd "$(dirname "$0")/../../.."
mkdir -p "$out"
key=(--model Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp --prompt board-2048
     --prompt-sha256 "$(sha256sum research/e2e/prompts/board-2048.txt | cut -d' ' -f1)" --decode eager
     --device "NVIDIA GeForce RTX 5090 Laptop GPU"
     --model-sha256 72ff9600aa2b0de77a5b27041a84448c2ce88c7b2055529fc23b3cd5bf518fd3 --window 12)
last=$(awk '/^[0-9]+[ \t]+[0-9]+/ { p = $1 } END { print p }' "$log")
awk -v pos="$last" '/^[0-9]+[ \t]+[0-9]+/ && $1 == pos {
        rest = $7; for (i = 8; i <= NF; i++) rest = rest " " $i
        printf "%-8s %-13s %-16s %-12s %-16.9e %-16s %s\n", $1, $2, $3, $4, $5 + 1.0, $6, rest; next }
    { print }' "$log" > "$out/red-1.log"
awk '/^[0-9]+[ \t]+[0-9]+/ && $1 == 2037 {
        printf "%-8s %-13s %-16s %-12s %-16s %-16s %s\n", $1, $2, $3, 99999, "5.000000000e-2", $6, "NO <-- FLIP"; next }
    { print }' "$log" > "$out/red-2.log"
for arm in green red-1 red-2; do
    in=$log; [ "$arm" = green ] || in=$out/$arm.log
    "$probe" --judge "$in" --max-flips 1 --floors tools/argmax-margin-floors.tsv "${key[@]}" \
        --stored-rows tools/argmax-margin-rows > "$out/$arm.txt" 2>&1
    echo "exit=$?" >> "$out/$arm.txt"
    echo "$arm: $(grep -E '^SUMMARY' "$out/$arm.txt" | cut -d' ' -f2-3) | $(grep -E '^STORED-ROWS' "$out/$arm.txt" | cut -c1-60) | $(tail -1 "$out/$arm.txt")"
done
