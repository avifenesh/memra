#!/usr/bin/env bash
# The G''' target-card build (DAY38 section 14, DAY39 section 5, DAY41 section 1), outside any hold, on the box's clone of
# this lane (/root/wt-a, never another lane's tree). From ONE tree, the tip: g3 (the tip as built), f1 (the tip plus
# rtx5090-day39/f1.patch, design F at one fill thread), hk (the tip plus rtx5090-day39/hk-revert-tip.patch, the day-32
# helper fill with design K), then the tip's test binaries; the tree checked back at the tip after each patched build.
# base (80039a8de) and gpp (G'', the day-38 sitting's tip b214bd2cf, the hump's control) are the day-38 sitting's binaries,
# copied with their hashes. usage: build.sh <tip_sha>
set -uo pipefail
R=/root/spill-receipts/a-g3
O=/root/spill-receipts/a-day38
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$R/bins/g3" "$R/bins/f1" "$R/bins/hk" "$R/bins/base" "$R/bins/gpp"
L=$R/build-steps.log
cp "$O/bins/base/memra-server" "$R/bins/base/memra-server" && cp "$O/bins/tip/memra-server" "$R/bins/gpp/memra-server" || { echo "rc=2 (day-38 binaries)" >> "$R/build.log"; exit 2; }
cd /root/wt-a || exit 1
# The day-38 diagnosis copied its scripts in untracked; the tip tracks them (hash-checked when copied), so they go first.
git clean -q -f research/spill-a-20260919/ >> "$L" 2>&1
git fetch -q origin lane/spill-a-20260919 >> "$L" 2>&1
git checkout -q -B lane-a-g3-tip "$1" >> "$L" 2>&1 || { echo "rc=2 (checkout tip)" >> "$R/build.log"; exit 2; }
TIP=$(git rev-parse HEAD); echo "$TIP" > "$R/tree-tip.sha"
clean() { [ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$TIP" ]; }
echo "== g3" >> "$L"
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (g3)" >> "$R/build.log"; exit 1; }
cp target/release/memra-server "$R/bins/g3/memra-server"
echo "== f1" >> "$L"
git apply research/spill-a-20260919/rtx5090-day39/f1.patch >> "$L" 2>&1 || { echo "rc=2 (f1 patch)" >> "$R/build.log"; exit 2; }
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1; brc=$?
cp target/release/memra-server "$R/bins/f1/memra-server"
git checkout -q -- crates docs tools; clean || { echo "rc=2 (tree after f1)" >> "$R/build.log"; exit 2; }
[ $brc -eq 0 ] || { echo "rc=1 (f1)" >> "$R/build.log"; exit 1; }
echo "== hk" >> "$L"
git apply -3 research/spill-a-20260919/rtx5090-day39/hk-revert-tip.patch >> "$L" 2>&1 || { echo "rc=2 (hk patch)" >> "$R/build.log"; exit 2; }
git diff --cached --stat > "$R/bins/hk/patch-applied.stat" 2>&1
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1; brc=$?
cp target/release/memra-server "$R/bins/hk/memra-server"
git reset -q; git checkout -q -- crates docs tools; clean || { echo "rc=2 (tree after hk)" >> "$R/build.log"; exit 2; }
[ $brc -eq 0 ] || { echo "rc=1 (hk)" >> "$R/build.log"; exit 1; }
echo "== the tip's test binaries" >> "$L"
nice -n 5 cargo test -p memra-server --lib --no-run >> "$L" 2>&1 || { echo "rc=1 (server tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-engine --lib --no-run >> "$L" 2>&1 || { echo "rc=1 (engine tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-tier --test contracts --no-run >> "$L" 2>&1 || { echo "rc=1 (tier tests)" >> "$R/build.log"; exit 1; }
sha256sum "$R"/bins/*/memra-server | tee "$R/binaries.sha256"
# Markers (grep -a reads the binary as text): the G''' capture wording, in g3, f1 and hk only.
for b in g3 f1 hk base gpp; do echo "$b door's-receipt-stream: $(grep -ac "door's receipt stream" "$R/bins/$b/memra-server")"; done > "$R/markers.txt"
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz)'; free -g | head -2; } > "$R/host-shape.txt" 2>&1
clean || { echo "rc=2 (tree at the end)" >> "$R/build.log"; exit 2; }
echo "rc=0" >> "$R/build.log"
