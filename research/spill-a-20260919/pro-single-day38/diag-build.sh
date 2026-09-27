#!/usr/bin/env bash
# DAY38 section 12, outside any hold: the diagnostic arm X2 = the sitting's tip with diag-one-stream.patch (the receipt
# stream made the copy stream: the receipt kernels queue ahead of the copies as under G, everything else G''), built into
# bins/x2 and the tree taken back to the tip. usage: diag-build.sh
set -uo pipefail
R=/root/spill-receipts/a-day38
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
TIP=$(cat "$R/tree-tip.sha")
P=research/spill-a-20260919/pro-single-day38/diag-one-stream.patch
mkdir -p "$R/bins/x2"
git apply "$P" >> "$R/diag-build.log" 2>&1 || { echo "rc=2 (patch)" >> "$R/diag-build.log"; exit 2; }
nice -n 5 cargo build --release -p memra-server >> "$R/diag-build.log" 2>&1; brc=$?
git diff --stat > "$R/bins/x2/patch-applied.stat"
git checkout -q -- crates
[ "$(git rev-parse HEAD)" = "$TIP" ] && git diff --quiet || { echo "rc=2 (tree not restored)" >> "$R/diag-build.log"; exit 2; }
[ $brc -eq 0 ] || { echo "rc=1 (build)" >> "$R/diag-build.log"; exit 1; }
cp target/release/memra-server "$R/bins/x2/memra-server"
sha256sum "$R/bins/x2/memra-server" > "$R/bins/x2/memra-server.sha256"
echo "rc=0" >> "$R/diag-build.log"
