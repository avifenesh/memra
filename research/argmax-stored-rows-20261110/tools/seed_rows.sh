#!/usr/bin/env bash
# seed_rows.sh <built argmax-margin-probe>
# Main's stored rows for every battery cell that has a floor row: the last 12 rows of the floor
# row's calibration receipt (main 031f146690's probe over the whole prompt), written by the
# probe's own `--judge <receipt> --calibrate --rows-out`. The model digest of each device's
# artifact comes from model-sha256.tsv beside this script. Writes tools/argmax-margin-rows/.
set -euo pipefail
probe=$(realpath "$1")
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here/../.."
out=tools/argmax-margin-rows
mkdir -p "$out"
commit=031f146690
# The battery's models: tools/release-roster.tsv, the file name without .gguf.
mapfile -t roster < <(grep -v '^#' tools/release-roster.tsv | awk -F'\t' 'NF >= 3 { n = split($3, p, "/"); sub(/\.gguf$/, "", p[n]); print p[n] }')
grep -v '^#' tools/argmax-margin-floors.tsv | while IFS=$'\t' read -r model prompt sha decode device _floor _rows _by _p50 _p99 _p999 _known receipt; do
    printf '%s\n' "${roster[@]}" | grep -qxF "$model" || continue
    digest=$(awk -F'\t' -v d="$device" -v m="$model" '$1 == d && $2 == m { print $3 }' "$here/model-sha256.tsv")
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "seed_rows: no model digest for $model on $device" >&2; exit 1; }
    slug=$(echo "$model-$decode-$device" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9.]\+/-/g; s/-$//')
    "$probe" --judge "$receipt" --calibrate --receipt "$receipt" --model "$model" --prompt "$prompt" \
        --prompt-sha256 "$sha" --decode "$decode" --device "$device" --model-sha256 "$digest" --window 12 \
        --rows-out "$out/$slug.rows" --commit "$commit" > /dev/null
    echo "$out/$slug.rows  <- $receipt"
done
