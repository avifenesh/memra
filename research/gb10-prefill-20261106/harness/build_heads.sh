#!/usr/bin/env bash
# Runs ON a GB10 host: build and bank the base head and the v2 PR code head.
# usage: build_heads.sh BASE_SHA V2_SHA   (V2 comes from v2pr.bundle, base from origin)
set -u
D=$HOME/.cache/memra-gemm-v2; T=$D/tree
B=$1; V=$2
git -C "$T" checkout -q -- . 2>/dev/null
git -C "$T" fetch -q origin main || exit 1
git -C "$T" fetch -q "$D/v2pr.bundle" research/gb10-f8f4-gemm-v2:refs/heads/gemmv2-pr || exit 1
for spec in "base=$B" "v2pr=$V"; do
  name=${spec%%=*}; sha=${spec#*=}
  if ls -d "$D/bins/$name-${sha:0:10}" "$D/bins/lanehead-${sha:0:10}" 2>/dev/null | grep -q .; then echo "have ${sha:0:10}"; continue; fi
  git -C "$T" checkout -q --detach "$sha" || { echo "checkout $sha failed"; exit 1; }
  git -C "$T" log --oneline -1
  JOBS=12 bash "$D/build.sh" "$name" || exit 1
done
echo HEADS-DONE
