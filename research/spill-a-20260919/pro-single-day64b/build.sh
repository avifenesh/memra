#!/usr/bin/env bash
# DAY64 (OWED item 18, the late promotes placed) target-card build, outside any hold, on the box's clone of this lane (/root/wt-a): the tip's
# release server (the on-tick lines built in). usage: build.sh <tip_sha>
set -uo pipefail
R=/root/spill-receipts/a-d64b
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$R/bins/tip"
L=$R/build-steps.log
[ -d /root/wt-a/.git ] || git clone -q --filter=blob:none https://github.com/avifenesh/memra.git /root/wt-a >> "$L" 2>&1
cd /root/wt-a || exit 1
git fetch -q origin lane/spill-a-20260919 >> "$L" 2>&1
git checkout -q -B lane-a-d64b "$1" >> "$L" 2>&1 || { echo "rc=2 (checkout tip)" >> "$R/build.log"; exit 2; }
git rev-parse HEAD > "$R/tree-tip.sha"
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (tip)" >> "$R/build.log"; exit 1; }
cp target/release/memra-server "$R/bins/tip/memra-server"
sha256sum "$R/bins/tip/memra-server" | tee "$R/binaries.sha256"
echo "timing wording: $(grep -ac 'span receipt: fill ' "$R/bins/tip/memra-server")" > "$R/markers.txt"
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz)'; free -g | head -2; } > "$R/host-shape.txt" 2>&1
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "rc=2 (tree at the end)" >> "$R/build.log"; exit 2; }
echo "rc=0" >> "$R/build.log"
