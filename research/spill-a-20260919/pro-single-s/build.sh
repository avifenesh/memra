#!/usr/bin/env bash
# The design-S target-card build (DAY40 section 3), outside any hold, on the box's clone of this lane (/root/wt-a): the S
# tip's release server (s) and its test binaries; g4 is the G4 sitting's (/root/spill-receipts/a-g4/bins/g4), copied with
# its hash. usage: build.sh <tip_sha>
set -uo pipefail
R=/root/spill-receipts/a-s
O=/root/spill-receipts/a-g4
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$R/bins/s" "$R/bins/g4"
L=$R/build-steps.log
cp "$O/bins/g4/memra-server" "$R/bins/g4/memra-server" || { echo "rc=2 (the G4 binary)" >> "$R/build.log"; exit 2; }
cd /root/wt-a || exit 1
git fetch -q origin lane/spill-a-20260919 >> "$L" 2>&1
git checkout -q -B lane-a-s-tip "$1" >> "$L" 2>&1 || { echo "rc=2 (checkout tip)" >> "$R/build.log"; exit 2; }
git rev-parse HEAD > "$R/tree-tip.sha"
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (s)" >> "$R/build.log"; exit 1; }
cp target/release/memra-server "$R/bins/s/memra-server"
nice -n 5 cargo test -p memra-server --lib --no-run >> "$L" 2>&1 || { echo "rc=1 (server tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-engine --lib --no-run >> "$L" 2>&1 || { echo "rc=1 (engine tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-tier --test contracts --no-run >> "$L" 2>&1 || { echo "rc=1 (tier tests)" >> "$R/build.log"; exit 1; }
sha256sum "$R"/bins/*/memra-server | tee "$R/binaries.sha256"
for b in s g4; do echo "$b span-receipt wording: $(grep -ac 'span landed bytes differ from' "$R/bins/$b/memra-server")"; done > "$R/markers.txt"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "rc=2 (tree at the end)" >> "$R/build.log"; exit 2; }
echo "rc=0" >> "$R/build.log"
