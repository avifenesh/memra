#!/usr/bin/env bash
# GB10 box: build hy4f-spec, hy4f-gpu, memra-server (sm_121a; CPU only, low weight)
set -o pipefail
. "$HOME/.cargo/env"
cd "$HOME/hy4flash/s11/src" || exit 2
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CARGO_TARGET_DIR=$HOME/hy4flash/s11/target
export CARGO_BUILD_JOBS=${JOBS:-16}
nice -n 10 cargo build --release -p memra-engine --bin hy4f-spec --bin hy4f-gpu -p memra-server --bin memra-server 2>&1
echo "BUILD_EXIT=$?"
