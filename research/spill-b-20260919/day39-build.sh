#!/usr/bin/env bash
# WP-B day 39 builds (DAY39.md 1.3): green = GREEN_SHA (the r4 tree, the lane tip at the first boot), red = the same tree with
# day39-red.patch applied (the reverse of that commit's crates diff: the day-33/35 booking), each in its own detached
# worktree, one after the other into one CARGO_TARGET_DIR. Records source.commit (and the patch sha256 for red) and
# SHA256SUMS. usage: day39-build.sh <bins dir>
# env: WT (the lane checkout), GREEN_SHA, TARGET (CARGO_TARGET_DIR), WRAP (a command prefix, e.g. the CPU quota scope)
set -uo pipefail
OUT=${1:?bins dir}
WT=${WT:-$HOME/projects/wt-spill-b}
GREEN_SHA=${GREEN_SHA:-c6f9282c26e71349c0263b88dc7beb59c4a57797}
TARGET=${TARGET:-$WT/target/day39-build}
PATCH=$WT/research/spill-b-20260919/day39-red.patch
read -r -a WRAP <<< "${WRAP:-nice -n 10}"
mkdir -p "$OUT/green" "$OUT/red"
build() { # <role>
  local role=$1 W=$WT/target/wt-b39-$1
  [ -x "$OUT/$role/memra-server" ] && return 0
  git -C "$WT" worktree add --detach "$W" "$GREEN_SHA" > "$OUT/$role/worktree.log" 2>&1 || { echo "worktree $role failed"; return 1; }
  if [ "$role" = red ]; then
    git -C "$W" apply "$PATCH" || { echo "red patch does not apply"; git -C "$WT" worktree remove --force "$W"; return 1; }
  fi
  ( cd "$W" && CARGO_TARGET_DIR=$TARGET "${WRAP[@]}" cargo build --release -p memra-server --bin memra-server ) > "$OUT/$role/build.log" 2>&1
  local rc=$?; echo "exit=$rc" >> "$OUT/$role/build.log"
  if [ $rc = 0 ]; then
    cp "$TARGET/release/memra-server" "$OUT/$role/memra-server"
    { git -C "$W" rev-parse HEAD; [ "$role" = red ] && sha256sum "$PATCH" | sed "s|$WT/||"; git -C "$W" diff --stat; } > "$OUT/$role/source.commit"
  fi
  git -C "$WT" worktree remove --force "$W" >> "$OUT/$role/worktree.log" 2>&1
  return $rc
}
build green || exit 1
build red || exit 1
( cd "$OUT" && sha256sum green/memra-server red/memra-server > SHA256SUMS )
cat "$OUT/SHA256SUMS"
