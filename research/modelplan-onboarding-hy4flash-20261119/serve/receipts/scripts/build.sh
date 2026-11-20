#!/usr/bin/env bash
# GB10 box: build memra-server, hy4f-gpu and memra for sm_121a (CPU only, low weight)
set -o pipefail
. "$HOME/.cargo/env"
cd "$HOME/hy4flash/pr9/src" || exit 2
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CARGO_TARGET_DIR=$HOME/hy4flash/pr9/target
export CARGO_BUILD_JOBS=${JOBS:-10}
nice -n 10 cargo build --release -p memra-engine --bin hy4f-gpu -p memra-cli --bin memra -p memra-server --bin memra-server 2>&1
echo "BUILD_EXIT=$?"
