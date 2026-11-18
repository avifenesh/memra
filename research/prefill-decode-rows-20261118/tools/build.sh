#!/usr/bin/env bash
# build.sh: build row-bisect + argmax-margin-probe for sm_121a in ~/.cache/memra-1203/tree.
set -uo pipefail
cd "$HOME/.cache/memra-1203/tree" || exit 2
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export MEMRA_CUDA_ARCH=121a CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
export CARGO_TARGET_DIR=$HOME/.cache/memra-1203/target
start=$(date +%s)
nice -n 10 cargo build --release --locked -j 16 -p memra-engine --bin row-bisect --bin argmax-margin-probe
rc=$?
echo "BUILD-RC $rc secs $(( $(date +%s) - start ))"
[ $rc -eq 0 ] && sha256sum "$CARGO_TARGET_DIR"/release/row-bisect "$CARGO_TARGET_DIR"/release/argmax-margin-probe
echo BUILD-DONE
