#!/usr/bin/env bash
# BOX3 day-32 build: from the shipped bundle, check out in /root/wt-a (my own worktree; the reference build /root/memra-spill
# is never touched) first the PRE-H2D binary's tree (the log-only owner-segment field, B2's baseline, DAY31 section 2) and then
# the H2D tree, building the release server at each and keeping both binaries; writes the receipt the driver waits for.
# Run only after lane B's chain has finished (a build is CPU-heavy). usage: build.sh <pre_sha> <h2d_sha>
set -uo pipefail
PRE=$1; SHA=$2
R=/root/spill-receipts/a-day32
mkdir -p "$R/bins"
cd /root/wt-a || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
git fetch /root/a32.bundle lane/spill-a-20260919 > "$R/fetch.log" 2>&1  # the lane tip travels as a git bundle; removed both ends at close
for pair in "pre:$PRE" "h2d:$SHA"; do
  name=${pair%%:*}; sha=${pair#*:}
  git checkout -q -B "lane-a-day32-$name" "$sha" >> "$R/fetch.log" 2>&1 || { echo "rc=2" >> "$R/build.log"; exit 2; }
  git rev-parse HEAD > "$R/tree-$name.sha"
  echo "== $name $sha" >> "$R/build-steps.log"
  nice -n 5 cargo build --release -p memra-server >> "$R/build-steps.log" 2>&1 || { echo "rc=1 ($name)" >> "$R/build.log"; exit 1; }
  cp target/release/memra-server "$R/bins/memra-server-$name"
  sha256sum "$R/bins/memra-server-$name" > "$R/bins/memra-server-$name.sha256"
done
# The H2D tree stays checked out: the gates, the scripts and the unit cells run from it.
git rev-parse HEAD > "$R/tree.sha"
echo "rc=0" >> "$R/build.log"
