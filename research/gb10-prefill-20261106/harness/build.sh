#!/usr/bin/env bash
# Build memra in the prefill lane worktree on the GB10 host.
# usage: build.sh TREE TARGET [extra cargo args]
set -euo pipefail
TREE=${1:?tree}
TARGET=${2:?target}
shift 2
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR="$TARGET"
cd "$TREE"
git log --oneline -1
date -Is
flock "$HOME/.cache/memra-rig-build.lock" nice -n 10 cargo build --release --locked \
  -p memra-engine --bin kernel-check --bin run-gen --bin run-spec \
  -p memra-server --bin memra-server "$@" 2>&1
date -Is
ls -la "$TARGET/release/memra-server" "$TARGET/release/run-gen" "$TARGET/release/kernel-check"
sha256sum "$TARGET/release/memra-server" "$TARGET/release/run-gen" "$TARGET/release/kernel-check"
echo BUILD-OK
