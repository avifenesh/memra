#!/usr/bin/env bash
# Day-38 target-card build (DAY38.md section 4), outside any hold: /root/wt-a is this lane's own clone (a blobless clone of
# origin when absent; never another lane's tree). Two release servers: the base (80039a8de, the lane tip before design G's
# code; its production code equals main's) and the tip (designs G' and P, DAY37's finding-5 fix), then the tip's test
# binaries. The port guard's tool (ss or lsof) checked, iproute2 installed when neither is present (day 34's attempt 1).
# Writes the receipt the driver waits for. usage: build.sh <tip_sha>
set -uo pipefail
R=/root/spill-receipts/a-day38
BASE=80039a8de
mkdir -p "$R/bins/tip" "$R/bins/base"
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
if ! command -v ss >/dev/null && ! command -v lsof >/dev/null; then
  (apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q iproute2) > "$R/apt-iproute2.log" 2>&1
fi
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "rc=3 (no ss or lsof for the port guard)" >> "$R/build.log"; exit 3; }
[ -d /root/wt-a/.git ] || git clone -q --filter=blob:none https://github.com/avifenesh/memra.git /root/wt-a > "$R/fetch.log" 2>&1
cd /root/wt-a || exit 1
git fetch -q origin lane/spill-a-20260919 >> "$R/fetch.log" 2>&1
echo "== base $BASE" >> "$R/build-steps.log"
git checkout -q -B lane-a-day38-base "$BASE" >> "$R/fetch.log" 2>&1 || { echo "rc=2 (checkout base)" >> "$R/build.log"; exit 2; }
git rev-parse HEAD > "$R/tree-base.sha"
nice -n 5 cargo build --release -p memra-server >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (base)" >> "$R/build.log"; exit 1; }
cp target/release/memra-server "$R/bins/base/memra-server"
sha256sum "$R/bins/base/memra-server" > "$R/bins/base/memra-server.sha256"
echo "== tip $1" >> "$R/build-steps.log"
git checkout -q -B lane-a-day38-tip "$1" >> "$R/fetch.log" 2>&1 || { echo "rc=2 (checkout tip)" >> "$R/build.log"; exit 2; }
git rev-parse HEAD > "$R/tree-tip.sha"
nice -n 5 cargo build --release -p memra-server >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (tip)" >> "$R/build.log"; exit 1; }
cp target/release/memra-server "$R/bins/tip/memra-server"
sha256sum "$R/bins/tip/memra-server" > "$R/bins/tip/memra-server.sha256"
# G' marker: the tip's receipt line exists in the tip binary only (grep -a reads the binary as text; no binutils needed).
grep -ac "receipts on the receipt stream" "$R/bins/tip/memra-server" > "$R/bins/tip/g-marker.count"
grep -ac "receipts on the receipt stream" "$R/bins/base/memra-server" > "$R/bins/base/g-marker.count"
echo "== tip test binaries" >> "$R/build-steps.log"
nice -n 5 cargo test -p memra-server --lib --no-run >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (server tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-engine --lib --no-run >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (engine tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-tier --test contracts --no-run >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (tier tests)" >> "$R/build.log"; exit 1; }
echo "== day39 fill survey probe" >> "$R/build-steps.log"
(cd research/spill-a-20260919/day39-fill-survey && nice -n 5 cargo build --release --target-dir "$R/fill-target") >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (fill probe)" >> "$R/build.log"; exit 1; }
sha256sum "$R/fill-target/release/day39-fill-survey" > "$R/fill-target/day39-fill-survey.sha256"
# The host's shape for DAY39 (item 3 decides per host CPU class): CPU model and counts, memory. No host name or id.
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz)'; free -g | head -2; } > "$R/host-shape.txt" 2>&1
git rev-parse HEAD > "$R/tree.sha"
echo "rc=0" >> "$R/build.log"
