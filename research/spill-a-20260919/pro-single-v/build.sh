#!/usr/bin/env bash
# The design-V target-card build (DAY47.md section 1), outside any hold, on the box's clone of this lane (/root/wt-a,
# never another lane's tree). From ONE clone: v (the tip as built) and base (the tip's crates taken to <base_sha>, the
# lane before V's code), then the tip's test binaries; the tree checked back at the tip after each.
# usage: build.sh <tip_sha> <base_sha>
set -uo pipefail
R=/root/spill-receipts/a-v
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$R/bins/v" "$R/bins/base"
L=$R/build-steps.log
cd /root/wt-a || exit 1
git fetch -q origin lane/spill-a-20260919 >> "$L" 2>&1
git checkout -q -B lane-a-v-tip "$1" >> "$L" 2>&1 || { echo "rc=2 (checkout tip)" >> "$R/build.log"; exit 2; }
TIP=$(git rev-parse HEAD); echo "$TIP" > "$R/tree-tip.sha"; echo "$2" > "$R/tree-base.sha"
clean() { [ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$TIP" ]; }
echo "== v" >> "$L"
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (v)" >> "$R/build.log"; exit 1; }
cp target/release/memra-server "$R/bins/v/memra-server"
for arm in base; do
  sha=$2
  echo "== $arm ($sha)" >> "$L"
  # The arm's crates exactly (files the tip added are removed too): the tip-to-arm crate diff, applied.
  git diff --binary "$TIP" "$sha" -- crates | git apply >> "$L" 2>&1 || { echo "rc=2 ($arm crates)" >> "$R/build.log"; exit 2; }
  nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1; brc=$?
  cp target/release/memra-server "$R/bins/$arm/memra-server"
  git checkout -q "$TIP" -- crates; git reset -q; git clean -q -fd crates; clean || { echo "rc=2 (tree after $arm)" >> "$R/build.log"; exit 2; }
  [ $brc -eq 0 ] || { echo "rc=1 ($arm)" >> "$R/build.log"; exit 1; }
done
echo "== the tip's test binaries" >> "$L"
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (v rebuild)" >> "$R/build.log"; exit 1; }
cmp -s target/release/memra-server "$R/bins/v/memra-server" || echo "note: the v rebuild differs in bytes" >> "$L"
nice -n 5 cargo test -p memra-server --lib --no-run >> "$L" 2>&1 || { echo "rc=1 (server tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-engine --lib --no-run >> "$L" 2>&1 || { echo "rc=1 (engine tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-tier --test contracts --no-run >> "$L" 2>&1 || { echo "rc=1 (tier tests)" >> "$R/build.log"; exit 1; }
sha256sum "$R"/bins/*/memra-server | tee "$R/binaries.sha256"
for b in v base; do echo "$b pause-off-tick wording: $(grep -ac 'demoting off the tick' "$R/bins/$b/memra-server")"; done > "$R/markers.txt"
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz)'; free -g | head -2; } > "$R/host-shape.txt" 2>&1
clean || { echo "rc=2 (tree at the end)" >> "$R/build.log"; exit 2; }
echo "rc=0" >> "$R/build.log"
