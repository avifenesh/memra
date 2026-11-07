#!/usr/bin/env bash
# Runs ON a GB10 host: after the base build, check out the lane head from the bundle and bank it.
# usage: build_cand.sh NAME BRANCH
set -u
D=$HOME/.cache/memra-attn-gb10
NAME=$1; BR=${2:-research/gb10-prefill-attn}
waitfor --grep 'BASE-BUILD-DONE' "$D/logs/build-base.out" --timeout 60m > /dev/null 2>&1
git -C "$D/tree" fetch -q "$D/h2p.bundle" "$BR:refs/heads/attn-lane" --force || exit 1
git -C "$D/tree" checkout -q --detach attn-lane || exit 1
git -C "$D/tree" log --oneline -1
JOBS=8 bash "$D/box/build.sh" "$NAME"
echo "CAND-BUILD-DONE $NAME"
