#!/usr/bin/env bash
# floor_rows.sh <built argmax-margin-probe>
# The floors-table row of every banked calibration cell, from its raw probe log, through the
# gate's own `--judge --calibrate`. Prints the rows (tab-separated, the table's column order)
# on stdout and a Markdown summary on stderr.
#
# A cell is calibration/<device>/<cell>/ with probe.log (the probe's raw output over the whole
# prompt) and result.txt (model, decode program, prompt sha256, device name, probe sha256).
set -uo pipefail
judge=$(realpath "$1")
cd "$(dirname "$0")/../../.."
lane=research/argmax-gate-rule-20261110
declare -A PROMPT_OF
for p in research/e2e/prompts/*.txt; do
    PROMPT_OF[$(sha256sum "$p" | cut -d' ' -f1)]=$(basename "${p%.txt}")
done
echo "| device | model | prompt | decode | rows | agreeing | flips | most flips in 12 rows | floor | set by row | probe sha256 | receipt |" >&2
echo "|---|---|---|---|---|---|---|---|---|---|---|---|" >&2
for res in "$lane"/calibration/*/*/result.txt; do
    dir=$(dirname "$res")
    grep -q '^rc=0' "$res" || { echo "floor_rows: $dir did not finish (no rc=0)" >&2; exit 1; }
    model=$(sed -n 's/^model=\(.*\)\.gguf model_bytes=.*/\1/p' "$res")
    arm=$(sed -n 's/^tag=.* arm=\(.*\)$/\1/p' "$res")
    sha=$(sed -n 's/^prompt_sha256=//p' "$res")
    gpu=$(sed -n 's/^gpu=//p' "$res")
    probe=$(sed -n 's/^probe_sha256=//p' "$res")
    prompt=${PROMPT_OF[$sha]:-}
    [ -n "$model" ] && [ -n "$arm" ] && [ -n "$gpu" ] && [ -n "$prompt" ] \
        || { echo "floor_rows: $res is missing a field or names an unknown prompt" >&2; exit 1; }
    out=$("$judge" --judge "$dir/probe.log" --calibrate --receipt "$dir/probe.log" --model "$model" \
        --prompt "$prompt" --prompt-sha256 "$sha" --decode "$arm" --device "$gpu") \
        || { echo "floor_rows: no floor from $dir: $out" >&2; exit 1; }
    echo "$out" | sed -n 's/^FLOOR-ROW: //p'
    cal=$(echo "$out" | sed -n 's/^CALIBRATION: //p')
    field() { echo "$cal" | tr ' ' '\n' | sed -n "s/^$1=//p"; }
    printf '| %s | %s | %s | %s | %s | %s | %s | %s | %.4f | %s | %s | `%s` |\n' "$gpu" "$model" "$prompt" "$arm" \
        "$(field rows)" "$(field agreeing)" "$(field flips)" "$(field max_flips_in_12)" "$(field floor)" \
        "$(field set_by_row)" "${probe:0:12}" "${dir#"$lane"/}/probe.log" >&2
done
