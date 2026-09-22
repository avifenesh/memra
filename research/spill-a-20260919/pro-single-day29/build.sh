#!/usr/bin/env bash
# BOX3 day-29 build: check out the lane tip in /root/wt-a (my own worktree; the reference build /root/memra-spill is
# never touched) from the shipped bundle and build the release server, writing the receipt the driver waits for.
# usage: build.sh <sha>
set -uo pipefail
SHA=$1
R=/root/spill-receipts/a-day29
mkdir -p "$R"
cd /root/wt-a || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
git fetch /root/a29.bundle lane/spill-a-20260919 > "$R/fetch.log" 2>&1  # the lane tip travels as a git bundle (the box has no route to GitHub); removed both ends at close
git checkout -q -B lane-a-day29 "$SHA" >> "$R/fetch.log" 2>&1 || { echo "rc=2" >> "$R/build.log"; exit 2; }
git rev-parse HEAD > "$R/tree.sha"
cargo build --release -p memra-server > "$R/build.log" 2>&1
echo "rc=$?" >> "$R/build.log"
