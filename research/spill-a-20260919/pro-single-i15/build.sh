#!/usr/bin/env bash
# OWED item 15's target-card build (DAY43.md section 1), outside any hold, on the box's clone of this lane (/root/wt-a).
# g4 = the S2 sitting's g4 (the lane before S2's code, b4816eda8's crates), copied with its hash once that build's
# receipt reads rc=0; g3 = the same crates with item15/g3-placement.patch (the crate diff 26676c037 -> 9ab5c1265: G4's
# placement change reversed, the receipt stream back), built from the tip's clone, the tree checked back at the tip.
# The survey probe (day38-hash-survey, detached from the workspace) is built too. usage: build.sh <tip_sha> <g4_sha>
set -uo pipefail
R=/root/spill-receipts/a-i15
S2=/root/spill-receipts/a-s2
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$R/bins/g4" "$R/bins/g3" "$R/bins/survey"
L=$R/build-steps.log
until grep -q '^rc=' "$S2/build.log" 2>/dev/null; do sleep 20; done
grep '^rc=' "$S2/build.log" | tail -1 | grep -q '^rc=0$' || { echo "rc=2 (the S2 build failed; no g4)" >> "$R/build.log"; exit 2; }
[ "$(cat "$S2/tree-g4.sha")" = "$2" ] || { echo "rc=2 (the S2 sitting's g4 is not $2)" >> "$R/build.log"; exit 2; }
cp "$S2/bins/g4/memra-server" "$R/bins/g4/memra-server" || { echo "rc=2 (copy g4)" >> "$R/build.log"; exit 2; }
cd /root/wt-a || exit 1
TIP=$(git rev-parse HEAD)
[ "$TIP" = "$(git rev-parse "$1")" ] || { echo "rc=2 (the clone is not at the tip)" >> "$R/build.log"; exit 2; }
echo "$TIP" > "$R/tree-tip.sha"; echo "$2" > "$R/tree-g4.sha"
clean() { [ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$TIP" ]; }
echo "== g3 ($2 + g3-placement.patch)" >> "$L"
git diff --binary "$TIP" "$2" -- crates | git apply >> "$L" 2>&1 || { echo "rc=2 (g4 crates)" >> "$R/build.log"; exit 2; }
git apply research/spill-a-20260919/item15/g3-placement.patch >> "$L" 2>&1 || { git checkout -q "$TIP" -- crates; git reset -q; git clean -q -fd crates; echo "rc=2 (g3 patch)" >> "$R/build.log"; exit 2; }
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1; brc=$?
cp target/release/memra-server "$R/bins/g3/memra-server"
git checkout -q "$TIP" -- crates; git reset -q; git clean -q -fd crates; clean || { echo "rc=2 (tree after g3)" >> "$R/build.log"; exit 2; }
[ $brc -eq 0 ] || { echo "rc=1 (g3)" >> "$R/build.log"; exit 1; }
echo "== the survey probe" >> "$L"
(cd research/spill-a-20260919/day38-hash-survey && CARGO_TARGET_DIR=/root/spill-receipts/a-i15/survey-target nice -n 5 cargo build --release) >> "$L" 2>&1 || { echo "rc=1 (survey)" >> "$R/build.log"; exit 1; }
cp /root/spill-receipts/a-i15/survey-target/release/day38-hash-survey "$R/bins/survey/day38-hash-survey"
rm -rf /root/spill-receipts/a-i15/survey-target
git checkout -q -- research/spill-a-20260919/day38-hash-survey 2>/dev/null; clean || { echo "rc=2 (tree at the end)" >> "$R/build.log"; exit 2; }
sha256sum "$R"/bins/*/* | tee "$R/binaries.sha256"
for b in g4 g3; do echo "$b copy-stream-receipt-line: $(grep -ac 'receipts on the copy stream' "$R/bins/$b/memra-server") receipt-stream wording: $(grep -ac 'receipts on the receipt stream' "$R/bins/$b/memra-server")"; done > "$R/markers.txt"
echo "rc=0" >> "$R/build.log"
