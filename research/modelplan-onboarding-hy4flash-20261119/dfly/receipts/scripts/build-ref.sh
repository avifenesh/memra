#!/usr/bin/env bash
# GB10 box: build the reference drafter runner (CPU)
set -o pipefail
. "$HOME/.cargo/env"
cd "$HOME/hy4flash/pr10/src" || exit 2
export CARGO_TARGET_DIR=$HOME/hy4flash/pr10/target
export CARGO_BUILD_JOBS=${JOBS:-6}
nice -n 10 cargo build --release -p memra-reference --bin hy4f_dfly 2>&1
echo "BUILD_EXIT=$?"
