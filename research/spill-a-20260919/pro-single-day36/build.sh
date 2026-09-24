#!/usr/bin/env bash
# Day-36 target-card build (DAY36.md sections 3 and 3a), outside any hold: /root/wt-a is this lane's own clone (a
# blobless clone of origin when absent; never another lane's tree). Two release servers from ONE tree: the tip (M' with the
# guard fix from review on #711, the day-36 instrument; also the price cell's and the gates' binary) and the base (the
# same tip with M' removed by base-revert.patch, applied here and taken back out after the build, so the two arms differ
# only in M'); then the tip's test binaries. The port guard's tool (ss or lsof) checked, iproute2 installed when neither
# is present (day 34's attempt 1). Writes the receipt the driver waits for. usage: build.sh <tip_sha>
set -uo pipefail
R=/root/spill-receipts/a-day36
mkdir -p "$R/bins/tip" "$R/bins/base"
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
if ! command -v ss >/dev/null && ! command -v lsof >/dev/null; then
  (apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q iproute2) > "$R/apt-iproute2.log" 2>&1
fi
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "rc=3 (no ss or lsof for the port guard)" >> "$R/build.log"; exit 3; }
[ -d /root/wt-a/.git ] || git clone -q --filter=blob:none https://github.com/avifenesh/memra.git /root/wt-a > "$R/fetch.log" 2>&1
cd /root/wt-a || exit 1
git fetch -q origin lane/spill-a-20260919 >> "$R/fetch.log" 2>&1
git checkout -q -B lane-a-day36-tip "$1" >> "$R/fetch.log" 2>&1 || { echo "rc=2 (checkout)" >> "$R/build.log"; exit 2; }
git rev-parse HEAD > "$R/tree-tip.sha"
P=research/spill-a-20260919/pro-single-day36/base-revert.patch
sha256sum "$P" > "$R/base-revert.patch.sha256"
echo "== tip $1" >> "$R/build-steps.log"
nice -n 5 cargo build --release -p memra-server >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (tip)" >> "$R/build.log"; exit 1; }
cp target/release/memra-server "$R/bins/tip/memra-server"
sha256sum "$R/bins/tip/memra-server" > "$R/bins/tip/memra-server.sha256"
echo "== base = tip + $P" >> "$R/build-steps.log"
git apply "$P" >> "$R/build-steps.log" 2>&1 || { echo "rc=2 (patch)" >> "$R/build.log"; exit 2; }
nice -n 5 cargo build --release -p memra-server >> "$R/build-steps.log" 2>&1; brc=$?
git diff --stat > "$R/base-applied.stat"
git checkout -q -- crates && git clean -fdq crates
[ "$(git rev-parse HEAD)" = "$(cat "$R/tree-tip.sha")" ] && git diff --quiet || { echo "rc=2 (tree not restored)" >> "$R/build.log"; exit 2; }
[ $brc -eq 0 ] || { echo "rc=1 (base)" >> "$R/build.log"; exit 1; }
cp target/release/memra-server "$R/bins/base/memra-server"
sha256sum "$R/bins/base/memra-server" > "$R/bins/base/memra-server.sha256"
echo "== tip test binaries" >> "$R/build-steps.log"
nice -n 5 cargo test -p memra-server --lib --no-run >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (server tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-engine --lib --no-run >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (engine tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-tier --test contracts --no-run >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (tier tests)" >> "$R/build.log"; exit 1; }
git rev-parse HEAD > "$R/tree.sha"
echo "rc=0" >> "$R/build.log"
