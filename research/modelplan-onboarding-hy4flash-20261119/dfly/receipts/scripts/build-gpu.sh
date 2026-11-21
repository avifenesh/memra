#!/usr/bin/env bash
# GB10 box: build hy4f-spec, hy4f-gpu, memra-server (sm_121a; CPU only, low weight)
set -o pipefail
. "$HOME/.cargo/env"
cd "$HOME/hy4flash/pr10/src" || exit 2
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CARGO_TARGET_DIR=$HOME/hy4flash/pr10/target
export CARGO_BUILD_JOBS=${JOBS:-8}
nice -n 10 cargo build --release -p memra-engine --bin hy4f-spec --bin hy4f-gpu ${SERVER:+-p memra-server --bin memra-server -p memra-cli --bin memra} 2>&1
echo "BUILD_EXIT=$?"
