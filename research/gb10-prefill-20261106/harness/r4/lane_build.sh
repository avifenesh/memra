#!/usr/bin/env bash
# Runs ON a GB10 host: check the lane bundle's head out in this round's tree and build it.
# usage: lane_build.sh NAME [BUNDLE]   (default ~/.cache/memra-gb10-r4/lane.bundle, ref HEAD)
set -u
NAME=$1
D=$HOME/.cache/memra-gb10-r4
BUNDLE=${2:-$D/lane.bundle}
cd "$D/tree" || exit 1
git fetch -q "$BUNDLE" "HEAD:refs/heads/r4-lane-$NAME" || { echo "BUNDLE-FETCH-FAIL"; exit 1; }
sha=$(git rev-parse "refs/heads/r4-lane-$NAME")
git checkout -q --detach "$sha" || exit 1
git log --oneline -1
JOBS=${JOBS:-16} bash "$D/box/build.sh" "$NAME"
