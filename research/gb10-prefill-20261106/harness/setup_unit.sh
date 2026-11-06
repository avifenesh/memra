#!/usr/bin/env bash
# Runs ON a GB10 host that does not have the lane yet: fetch the lane bundle into the local memra
# repo, add the lane worktree, seed a private target dir from a warm one, then build+bank the
# named branches with build_queue.sh. usage: setup_unit.sh REPO WARM_TARGET BUNDLE name=branch ...
set -u
REPO=$1; WARM=$2; BUNDLE=$3; shift 3
D=$HOME/.cache/memra-prefill
mkdir -p "$D/bins" "$D/runs"
cd "$REPO" || exit 1
git fetch -q "$BUNDLE" 'refs/heads/research/*:refs/heads/research/*' 'refs/tags/gb10pf-main:refs/tags/gb10pf-main' || exit 1
git for-each-ref refs/heads/research/gb10-prefill-* refs/tags/gb10pf-main
[ -d "$D/tree" ] || git worktree add -q --detach "$D/tree" gb10pf-main || exit 1
[ -d "$D/target" ] || cp -a "$WARM" "$D/target"
bash "$D/build_queue.sh" "$@"
