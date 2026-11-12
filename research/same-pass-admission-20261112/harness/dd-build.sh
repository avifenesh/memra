#!/usr/bin/env bash
# Dev desk (memra#1255): memra-server for sm_120a. usage: dd-build.sh ARM...  (ARM names tree-ARM/)
# Each ARM builds tree-ARM/ into target-ARM/ and copies bin/memra-server-ARM-120a. Own scratch dir only.
set -u
D=$HOME/.cache/memra-1255
mkdir -p "$D/bin" "$D/logs"
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=${ARCH:-120a}
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
echo "$$" >> "$D/pids.txt"
for ARM in "$@"; do
  T=$D/tree-$ARM; export CARGO_TARGET_DIR=$D/target-$ARM-$MEMRA_CUDA_ARCH
  cd "$T" || exit 1
  echo "build $ARM $(date -Is) worker $(sha256sum crates/memra-server/src/worker.rs | cut -c1-16)"
  nice -n 10 cargo build --release -j 14 -p memra-server --bin memra-server > "$D/logs/build-$ARM-$MEMRA_CUDA_ARCH.log" 2>&1
  rc=$?
  grep -E "Compiling memra-server|Finished" "$D/logs/build-$ARM-$MEMRA_CUDA_ARCH.log" | tail -n 3
  if [ $rc = 0 ]; then
    cp -a "$CARGO_TARGET_DIR/release/memra-server" "$D/bin/memra-server-$ARM-$MEMRA_CUDA_ARCH"
    sha256sum "$D/bin/memra-server-$ARM-$MEMRA_CUDA_ARCH"
    strings "$D/bin/memra-server-$ARM-$MEMRA_CUDA_ARCH" | grep -o -m1 'memra-0\.[0-9]*\.[0-9]*-[0-9a-f]\{12\}'
  else
    echo "BUILD-FAILED $ARM"; grep -B2 -A12 '^error' "$D/logs/build-$ARM-$MEMRA_CUDA_ARCH.log" | head -60
  fi
done
echo "BUILD-DONE $(date -Is)"
