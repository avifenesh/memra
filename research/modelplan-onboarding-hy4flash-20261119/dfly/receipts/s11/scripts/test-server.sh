#!/usr/bin/env bash
set -o pipefail
. "$HOME/.cargo/env"
cd "$HOME/hy4flash/s11/src" || exit 2
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CARGO_TARGET_DIR=$HOME/hy4flash/s11/target
export CARGO_BUILD_JOBS=16
nice -n 10 cargo test --release -p memra-server --lib ${1:-hy4f_serve} 2>&1 | tail -n 40
echo "TEST_EXIT=$?"
