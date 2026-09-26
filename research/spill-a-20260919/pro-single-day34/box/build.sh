#!/usr/bin/env bash
# BOX4 day-34 build (DAY34.md section 4): /root/wt-a is this lane's own clone (a blobless clone of origin when absent;
# never another lane's tree), three release servers: the day-32 H2D binary, the day-33 binary (design F and the
# timeline), the day-34 binary (the lane tip). Writes the receipt the driver waits for. Run after LANE-B-680-DONE.
# usage: build.sh <d32_sha> <d33_sha> <d34_sha>
set -uo pipefail
R=/root/spill-receipts/a-day34
mkdir -p "$R/bins"
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
[ -d /root/wt-a/.git ] || git clone -q --filter=blob:none https://github.com/avifenesh/memra.git /root/wt-a > "$R/fetch.log" 2>&1
cd /root/wt-a || exit 1
git fetch -q origin lane/spill-a-20260919 >> "$R/fetch.log" 2>&1
for pair in "d32:$1" "d33:$2" "d34:$3"; do
  name=${pair%%:*}; sha=${pair#*:}
  git checkout -q -B "lane-a-day34-$name" "$sha" >> "$R/fetch.log" 2>&1 || { echo "rc=2 ($name)" >> "$R/build.log"; exit 2; }
  git rev-parse HEAD > "$R/tree-$name.sha"
  echo "== $name $sha" >> "$R/build-steps.log"
  nice -n 5 cargo build --release -p memra-server >> "$R/build-steps.log" 2>&1 || { echo "rc=1 ($name)" >> "$R/build.log"; exit 1; }
  cp target/release/memra-server "$R/bins/memra-server-$name"
  sha256sum "$R/bins/memra-server-$name" > "$R/bins/memra-server-$name.sha256"
done
# The day-34 tree stays checked out: the gates, the scripts and the unit cells run from it.
git rev-parse HEAD > "$R/tree.sha"
echo "rc=0" >> "$R/build.log"
