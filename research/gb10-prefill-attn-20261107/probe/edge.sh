#!/usr/bin/env bash
# edge-shape byte compare at ragged T and T_kv. usage: edge.sh TAG KERNELS (first = reference)
set -u
D=$HOME/.cache/memra-attn-gb10
TAG=${1:-edge}; KS=${2:-h2,ppad}
cd "$D/probe" || exit 1
{
for spec in "128 128,129,160,161" "130 130,4226" "192 192,4097" "1000 1000,1031,33000" "1024 8224,131070" "4033 4033,4097,70001" "4075 4075,8171,131051" "4096 4096,262144"; do
  set -- $spec
  ./probe_attn "$1" "$2" 1 "$KS" | grep -E "^T "
done
} > "$D/logs/$TAG.log" 2>&1
echo EDGE-DONE >> "$D/logs/$TAG.log"
