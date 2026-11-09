#!/usr/bin/env bash
# memra#1093 cold-long K A/B on wrx: build memra-server at main 1273a16f07 in my own dir on the root
# disk, with the kit's toolchain, on cores 24-31/56-63.
set -u
K=/home/access/memra-0140
D=/home/access/memra-1093r
export RUSTUP_HOME=$K/rustup CARGO_HOME=$K/cargo RUSTUP_TOOLCHAIN=1.97.1
export PATH=$K/cargo/bin:/usr/local/cuda/bin:/home/access/.local/bin:$PATH
export TMPDIR=$D/tmp
mkdir -p "$TMPDIR"
cd "$D/repo" || exit 1
git checkout -q --detach 1273a16f0749bc65a9a681c122657a883e6ce79c || exit 1
echo "head $(git rev-parse HEAD) $(date -Is)"
MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$D/target CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_LTO=fat \
  taskset -c 24-31,56-63 nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server
rc=$?
echo "build exit $rc $(date -Is)"
if [ $rc -eq 0 ]; then
  cp "$D/target/release/memra-server" "$D/memra-server-1273a16f07"
  sha256sum "$D/memra-server-1273a16f07"
fi
echo BUILD_DONE
