#!/usr/bin/env bash
# Day-36 target-card build (DAY36.md section 3), outside any hold: /root/wt-a is this lane's own clone (a blobless clone
# of origin when absent; never another lane's tree); three release servers (base, M', the tip) and the tip's test
# binaries; the port guard's tool (ss or lsof) checked, iproute2 installed when neither is present (day 34's attempt 1).
# Writes the receipt the driver waits for. usage: build.sh <base_sha> <m2_sha> <tip_sha>
set -uo pipefail
R=/root/spill-receipts/a-day36
mkdir -p "$R/bins"
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
if ! command -v ss >/dev/null && ! command -v lsof >/dev/null; then
  (apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q iproute2) > "$R/apt-iproute2.log" 2>&1
fi
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "rc=3 (no ss or lsof for the port guard)" >> "$R/build.log"; exit 3; }
[ -d /root/wt-a/.git ] || git clone -q --filter=blob:none https://github.com/avifenesh/memra.git /root/wt-a > "$R/fetch.log" 2>&1
cd /root/wt-a || exit 1
git fetch -q origin lane/spill-a-20260919 >> "$R/fetch.log" 2>&1
for pair in "base:$1" "m2:$2" "tip:$3"; do
  name=${pair%%:*}; sha=${pair#*:}
  git checkout -q -B "lane-a-day36-$name" "$sha" >> "$R/fetch.log" 2>&1 || { echo "rc=2 ($name)" >> "$R/build.log"; exit 2; }
  git rev-parse HEAD > "$R/tree-$name.sha"
  echo "== $name $sha" >> "$R/build-steps.log"
  nice -n 5 cargo build --release -p memra-server >> "$R/build-steps.log" 2>&1 || { echo "rc=1 ($name)" >> "$R/build.log"; exit 1; }
  mkdir -p "$R/bins/$name"
  cp target/release/memra-server "$R/bins/$name/memra-server"
  sha256sum "$R/bins/$name/memra-server" > "$R/bins/$name/memra-server.sha256"
done
# The tip stays checked out: the scripts, the gates and the unit cells run from it; its test binaries are built here.
echo "== tip test binaries" >> "$R/build-steps.log"
nice -n 5 cargo test -p memra-server --lib --no-run >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (server tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-engine --lib --no-run >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (engine tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-tier --test contracts --no-run >> "$R/build-steps.log" 2>&1 || { echo "rc=1 (tier tests)" >> "$R/build.log"; exit 1; }
git rev-parse HEAD > "$R/tree.sha"
echo "rc=0" >> "$R/build.log"
