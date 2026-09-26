#!/usr/bin/env bash
# Target-card day-33 build: from the shipped bundle, check out in /root/wt-a (my own worktree; the reference build /root/memra-spill
# is never touched) first the day-32 H2D binary's tree (DAY33's before arm) and then
# the day-33 tree, building the release server at each and keeping both binaries; writes the receipt the driver waits for.
# Run only when the box is free of other lanes. usage: build.sh <day32_sha> <day33_sha>
# Binary names kept from day 32 so the driver and doublepark-pair read the same paths: `pre` is the day-32 H2D binary
# (the before arm), `h2d` the day-33 binary (the after arm).
set -uo pipefail
PRE=$1; SHA=$2
R=/root/spill-receipts/a-day33
mkdir -p "$R/bins"
cd /root/wt-a || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
git fetch /root/a33.bundle lane/spill-a-20260919 > "$R/fetch.log" 2>&1  # the lane tip travels as a git bundle; removed both ends at close
for pair in "pre:$PRE" "h2d:$SHA"; do
  name=${pair%%:*}; sha=${pair#*:}
  git checkout -q -B "lane-a-day33-$name" "$sha" >> "$R/fetch.log" 2>&1 || { echo "rc=2" >> "$R/build.log"; exit 2; }
  git rev-parse HEAD > "$R/tree-$name.sha"
  echo "== $name $sha" >> "$R/build-steps.log"
  nice -n 5 cargo build --release -p memra-server >> "$R/build-steps.log" 2>&1 || { echo "rc=1 ($name)" >> "$R/build.log"; exit 1; }
  cp target/release/memra-server "$R/bins/memra-server-$name"
  sha256sum "$R/bins/memra-server-$name" > "$R/bins/memra-server-$name.sha256"
done
# The H2D tree stays checked out: the gates, the scripts and the unit cells run from it.
git rev-parse HEAD > "$R/tree.sha"
echo "rc=0" >> "$R/build.log"
