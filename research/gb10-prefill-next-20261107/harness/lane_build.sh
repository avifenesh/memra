#!/usr/bin/env bash
# Runs ON a GB10 host: check the lane bundle's head out in this lane's tree and build it.
# usage: lane_build.sh NAME   (bundle at ~/.cache/memra-gb10-next/lane.bundle, ref HEAD)
set -u
NAME=$1
D=$HOME/.cache/memra-gb10-next
cd "$D/tree" || exit 1
git fetch -q "$D/lane.bundle" "HEAD:refs/heads/gdn-lane-$NAME" || { echo "BUNDLE-FETCH-FAIL"; exit 1; }
sha=$(git rev-parse "refs/heads/gdn-lane-$NAME")
git checkout -q --detach "$sha" || exit 1
git log --oneline -1
JOBS=${JOBS:-8} bash "$D/box/build.sh" "$NAME"
