#!/usr/bin/env bash
# Runs ON a GB10 host: this lane's worktree (detached at origin/main) + a private warm target dir.
# usage: setup_tree.sh REPO WARM_TARGET
set -u
REPO=$1; WARM=$2
D=$HOME/.cache/memra-attn3
mkdir -p "$D/bins" "$D/runs" "$D/logs"
git -C "$REPO" fetch -q origin main || exit 1
[ -d "$D/tree" ] || git -C "$REPO" worktree add -q --detach "$D/tree" origin/main || exit 1
git -C "$D/tree" log --oneline -1
[ -d "$D/target" ] || cp -a "$WARM" "$D/target"
du -sh "$D/target"
echo SETUP-OK
