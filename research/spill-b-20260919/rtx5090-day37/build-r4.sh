#!/usr/bin/env bash
# DAY37 addendum E (r4) local builds, one source for the three days: memra-server and kv-tier-gate from R4_SHA in the
# lane checkout (refused unless crates/, Cargo.toml and Cargo.lock equal R4_SHA's and are clean), copied to target/day37/r4 (the DAY37 lane binary),
# target/day38/green and target/day39/green (the same file); then each red arm in a detached worktree at R4_SHA with its
# own patch (day38-red.patch, day39-red.patch). Records source.commit, the patch sha256 and SHA256SUMS per day.
# Every cargo call runs under the rig's CPU quota scope.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R4_SHA=${R4_SHA:?the r4 commit}
Q=(systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G nice -n 10)
cd "$WT" || exit 1
git diff --quiet "$R4_SHA" HEAD -- crates Cargo.toml Cargo.lock || { echo "the build inputs differ from $R4_SHA"; exit 1; }
[ -z "$(git status --short crates Cargo.toml Cargo.lock)" ] || { echo "the build inputs are dirty"; exit 1; }
"${Q[@]}" cargo build --release -p memra-server --bin memra-server -p memra-engine --bin kv-tier-gate > target/build-r4.log 2>&1 \
  || { echo "lane build failed"; exit 1; }
rm -rf target/day37/r4 target/day38 target/day39
mkdir -p target/day37/r4 target/day38/green target/day38/red target/day39/green target/day39/red
cp target/release/memra-server target/release/kv-tier-gate target/day37/r4/
for d in target/day37/r4 target/day38/green target/day39/green; do
  [ "$d" = target/day37/r4 ] || cp target/release/memra-server "$d/"
  git rev-parse "$R4_SHA" > "$d/source.commit"   # one line: gates.sh runs serve-smoke at it
  echo "built in the lane checkout at $(git rev-parse HEAD), build inputs equal to $R4_SHA" > "$d/source.note"
done
red() { # <day> <patch>
  local W=$WT/target/wt-b-r4-red$1 T=$WT/target/day39-build
  git worktree add --detach "$W" "$R4_SHA" > "target/day$1/red/worktree.log" 2>&1 || return 1
  git -C "$W" apply "$D/$2" || { git worktree remove --force "$W"; return 1; }
  ( cd "$W" && CARGO_TARGET_DIR=$T "${Q[@]}" cargo build --release -p memra-server --bin memra-server ) > "target/day$1/red/build.log" 2>&1
  local rc=$?
  [ $rc = 0 ] && cp "$T/release/memra-server" "target/day$1/red/memra-server"
  { git -C "$W" rev-parse HEAD; sha256sum "$D/$2" | sed "s|$WT/||"; git -C "$W" diff --stat; } > "target/day$1/red/source.commit"
  git worktree remove --force "$W" >> "target/day$1/red/worktree.log" 2>&1
  return $rc
}
red 38 day38-red.patch || { echo "day 38 red build failed"; exit 1; }
red 39 day39-red.patch || { echo "day 39 red build failed"; exit 1; }
sha256sum target/day37/r4/memra-server target/day37/r4/kv-tier-gate > "$D/rtx5090-day37/r4-binaries.sha256"
sha256sum target/day38/green/memra-server target/day38/red/memra-server > "$D/rtx5090-day38/binaries.sha256"
( cd target/day39 && sha256sum green/memra-server red/memra-server > SHA256SUMS )
cp target/day38/red/source.commit "$D/rtx5090-day38/red.source"; cp target/day39/red/source.commit "$D/rtx5090-day39/red.source"
echo "$R4_SHA" > "$D/rtx5090-day37/r4-binaries.source"
cat "$D/rtx5090-day37/r4-binaries.sha256" "$D/rtx5090-day38/binaries.sha256" target/day39/SHA256SUMS
