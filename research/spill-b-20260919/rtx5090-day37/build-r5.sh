#!/usr/bin/env bash
# DAY37 addendum H (r5) local builds: memra-server and kv-tier-gate together from R5_SHA in the lane checkout (refused
# unless crates/, Cargo.toml and Cargo.lock equal R5_SHA's and are clean), copied to target/day37/r5; the main arm from
# MAIN_SHA in its own worktree (build-arms.sh) into target/day37/r5main/main. Records source.commit and the sums.
# Every cargo call runs at nice 19 under the rig's 600% quota (the lead's rule while other lanes time cells), logged.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R5_SHA=${R5_SHA:-02dbdfa408a5d0215248c0ec2849cd599ca64083}
MAIN_SHA=${MAIN_SHA:-2c5edcb4cbf5}
W="systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19"
cd "$WT" || exit 1
git diff --quiet "$R5_SHA" HEAD -- crates Cargo.toml Cargo.lock || { echo "the build inputs differ from $R5_SHA"; exit 1; }
[ -z "$(git status --short crates Cargo.toml Cargo.lock)" ] || { echo "the build inputs are dirty"; exit 1; }
echo "$(date -u +%FT%TZ) WP-B day37 r5 build (nice 19, CPUQuota=600%)" >> "$D/cpu-concurrency.log"
if [ ! -x target/day37/r5/kv-tier-gate ]; then
  $W cargo build --release -p memra-server --bin memra-server -p memra-engine --bin kv-tier-gate > target/build-r5.log 2>&1 \
    || { echo "lane build failed"; exit 1; }
  mkdir -p target/day37/r5
  cp target/release/memra-server target/release/kv-tier-gate target/day37/r5/
  git rev-parse "$R5_SHA" > target/day37/r5/source.commit   # one line: gates.sh runs serve-smoke at it
  echo "built in the lane checkout at $(git rev-parse HEAD), build inputs equal to $R5_SHA" > target/day37/r5/source.note
fi
export WT; TARGET=$WT/target/arms-build WRAP="$W" bash "$D/build-arms.sh" "$WT/target/day37/r5main" "$MAIN_SHA" main \
  > "$D/rtx5090-day37/r5main-build.out" 2>&1 || { echo "main build failed"; exit 1; }
sha256sum target/day37/r5/memra-server target/day37/r5/kv-tier-gate target/day37/r5main/main/memra-server \
  > "$D/rtx5090-day37/r5-binaries.sha256"
{ echo "lane $R5_SHA"; echo "main $(git rev-parse "$MAIN_SHA")"; } > "$D/rtx5090-day37/r5-binaries.source"
cat "$D/rtx5090-day37/r5-binaries.sha256"
