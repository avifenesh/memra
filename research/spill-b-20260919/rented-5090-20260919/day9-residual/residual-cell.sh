#!/usr/bin/env bash
# One collector invocation holds the canonical GPU lock across both diagnostic sizes.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo 'usage: residual-cell.sh <native-binary> <artifact> <existing-output-directory>' >&2
    exit 2
fi
binary=$1
artifact=$2
out=$3
git rev-parse HEAD > "$out/source.commit"
sha256sum "$binary" > "$out/binary.sha256"
for context in 32768 16384; do
    printf 'RESIDUAL_DIAGNOSTIC context=%s start\n' "$context"
    "$binary" --artifact "$artifact" --case active --context "$context" \
        --tiers host --same-program --kv-allocator vmm --reclaim-diagnostic \
        --out "$out/receipt-$context"
    printf 'RESIDUAL_DIAGNOSTIC context=%s complete\n' "$context"
done
