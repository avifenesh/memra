#!/usr/bin/env bash
# WP-A day 45 (DAY45.md section 1, OWED item 16) on the local rig, outside any hold: the four release servers of DAY38
# section 20's cell, from one scratch worktree and one scratch target on disk (removed at the end): base 80039a8de,
# g4 26676c037 (the crates of section 19's g4 binary), g3 9ab5c1265 (G'''), gpp 358749c9f (G''). Every cargo step under
# the CPU quota. usage: build.sh <out>
set -uo pipefail
OUT=$1
cd "$(dirname "$0")/../../.." || exit 1
q() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G nice "$@"; }
SCR=$HOME/.cache/wt-a-d45
rm -rf "$SCR"; mkdir -p "$SCR"
git worktree add -q --detach "$SCR/wt" 80039a8de || exit 1
for pair in base:80039a8de g4:26676c037 g3:9ab5c1265 gpp:358749c9f; do
    arm=${pair%%:*}; sha=${pair##*:}
    mkdir -p "$OUT/$arm"
    (cd "$SCR/wt" && git checkout -q --detach "$sha" && q env CARGO_TARGET_DIR="$SCR/target" cargo build --release -p memra-server 2>&1 | tail -1)
    cp "$SCR/target/release/memra-server" "$OUT/$arm/memra-server" || exit 1
    echo "$sha" > "$OUT/$arm/tree.sha"
done
git worktree remove --force "$SCR/wt"
rm -rf "$SCR"
sha256sum "$OUT"/*/memra-server | tee "$OUT/binaries.sha256"
for b in base g4 g3 gpp; do echo "$b copy-stream receipt lines: $(grep -ac 'receipts on the copy stream' "$OUT/$b/memra-server") receipt-stream wording: $(grep -ac 'receipts on the receipt stream' "$OUT/$b/memra-server")"; done | tee "$OUT/markers.txt"
echo BUILD-DONE
