#!/usr/bin/env bash
# memra#1093 flip candidate on the RTX PRO 6000 box: build one commit (arg 1) on the cell's cores.
set -u
K=/home/access/memra-0140
D=/home/access/memra-1093r
c=$1
export RUSTUP_HOME=$K/rustup CARGO_HOME=$K/cargo RUSTUP_TOOLCHAIN=1.97.1
export PATH=$K/cargo/bin:/usr/local/cuda/bin:/home/access/.local/bin:$PATH
export TMPDIR=$D/tmp
cd "$D/repo" || exit 1
git checkout -q --detach "$c" || exit 1
echo "build $c $(date -Is)"
MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$D/target CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_LTO=fat \
  taskset -c 24-31,56-63 nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server \
  || { echo "build $c failed"; exit 1; }
cp "$D/target/release/memra-server" "$D/memra-server-$c"
sha256sum "$D/memra-server-$c"
echo BUILD4_DONE
