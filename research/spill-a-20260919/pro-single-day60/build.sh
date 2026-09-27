#!/usr/bin/env bash
# DAY60 (OWED item 11) target-card build, outside any hold: arm X the lane tip in /root/wt-a, arm Y the day-16 tree
# 1646d421b in its own worktree /root/wt-a-day16; each writes build-{x,y}.log with a final rc= line, the shape C's
# day29-box-run.sh waits for. usage: build.sh <tip_sha> <receipts_root>
set -uo pipefail
TIP=$1; R=$2
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$R"
[ -d /root/wt-a/.git ] || git clone -q --filter=blob:none https://github.com/avifenesh/memra.git /root/wt-a >> "$R/build-x.log" 2>&1
cd /root/wt-a || exit 1
git fetch -q origin lane/spill-a-20260919 >> "$R/build-x.log" 2>&1
git checkout -q -B lane-a-d60 "$TIP" >> "$R/build-x.log" 2>&1 || { echo "rc=2 (checkout tip)" >> "$R/build-x.log"; exit 2; }
git rev-parse HEAD > "$R/tree-x.sha"
nice -n 5 cargo build --release -p memra-server >> "$R/build-x.log" 2>&1; echo "rc=$?" >> "$R/build-x.log"
rm -rf /root/wt-a-day16
git worktree add -q --detach /root/wt-a-day16 1646d421b >> "$R/build-y.log" 2>&1 || { echo "rc=2 (worktree y)" >> "$R/build-y.log"; exit 2; }
git -C /root/wt-a-day16 rev-parse HEAD > "$R/tree-y.sha"
(cd /root/wt-a-day16 && nice -n 5 cargo build --release -p memra-server) >> "$R/build-y.log" 2>&1; echo "rc=$?" >> "$R/build-y.log"
sha256sum /root/wt-a/target/release/memra-server /root/wt-a-day16/target/release/memra-server > "$R/binaries.sha256"
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz)'; free -g | head -2; } > "$R/host-shape.txt" 2>&1
