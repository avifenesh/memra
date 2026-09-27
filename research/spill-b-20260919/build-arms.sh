#!/usr/bin/env bash
# WP-B arm builds (DAY38 addendum D, DAY39 addendum B, 2026-09-25): each arm is SHA plus an ordered list of this
# lane's patches, built in its own detached worktree into one CARGO_TARGET_DIR, one after another. An arm whose binary
# already exists is skipped. Records <bins>/<arm>/source.commit (one line, the SHA), source.patches (each patch's
# sha256) and <bins>/SHA256SUMS.
# usage: build-arms.sh <bins dir> <sha> <arm>[:<patch>[+<patch>...]] ...
# env: WT (the lane checkout; patches are read from its research dir), TARGET (CARGO_TARGET_DIR), WRAP (a prefix).
set -uo pipefail
OUT=${1:?bins dir}; SHA=${2:?sha}; shift 2
WT=${WT:-$HOME/projects/wt-spill-b}
TARGET=${TARGET:-$WT/target/arms-build}
D=$WT/research/spill-b-20260919
read -r -a WRAP <<< "${WRAP:-nice -n 10}"
SHA=$(git -C "$WT" rev-parse "$SHA") || { echo "unknown sha"; exit 1; }
for spec in "$@"; do
  arm=${spec%%:*}; patches=""; [ "$spec" != "$arm" ] && patches=${spec#*:}
  mkdir -p "$OUT/$arm"
  [ -x "$OUT/$arm/memra-server" ] && { echo "arm $arm: present"; continue; }
  W=$WT/target/wt-arm-$arm
  git -C "$WT" worktree add --detach "$W" "$SHA" > "$OUT/$arm/worktree.log" 2>&1 || { echo "arm $arm: worktree failed"; exit 1; }
  : > "$OUT/$arm/source.patches"
  IFS=+ read -r -a plist <<< "$patches"
  for p in "${plist[@]}"; do
    [ -n "$p" ] || continue
    git -C "$W" apply "$D/$p" || { echo "arm $arm: $p does not apply"; git -C "$WT" worktree remove --force "$W"; exit 1; }
    sha256sum "$D/$p" | sed "s|$WT/||" >> "$OUT/$arm/source.patches"
  done
  ( cd "$W" && CARGO_TARGET_DIR=$TARGET "${WRAP[@]}" cargo build --release -p memra-server --bin memra-server ) > "$OUT/$arm/build.log" 2>&1
  rc=$?; echo "exit=$rc" >> "$OUT/$arm/build.log"
  [ $rc = 0 ] && cp "$TARGET/release/memra-server" "$OUT/$arm/memra-server"
  echo "$SHA" > "$OUT/$arm/source.commit"
  git -C "$W" diff --stat >> "$OUT/$arm/source.patches"
  git -C "$WT" worktree remove --force "$W" >> "$OUT/$arm/worktree.log" 2>&1
  [ $rc = 0 ] || { echo "arm $arm: build failed"; exit 1; }
  echo "arm $arm: built"
done
( cd "$OUT" && sha256sum ./*/memra-server > SHA256SUMS )
cat "$OUT/SHA256SUMS"
