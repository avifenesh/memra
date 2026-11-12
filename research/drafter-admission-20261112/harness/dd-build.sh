#!/usr/bin/env bash
# Dev desk (memra#1173): memra-server for sm_120a. usage: dd-build.sh ARM...  (ARM = main | fix)
# main builds tree-main/ (target-main/), fix builds tree/ (target/). Own scratch dir only.
set -u
D=$HOME/.cache/memra-1173
mkdir -p "$D/bin" "$D/logs"
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
echo "$$" >> "$D/pids.txt"
for ARM in "$@"; do
  if [ "$ARM" = fix ]; then T=$D/tree; export CARGO_TARGET_DIR=$D/target
  else T=$D/tree-$ARM; export CARGO_TARGET_DIR=$D/target-$ARM; fi
  cd "$T" || exit 1
  echo "build $ARM $(date -Is) worker $(sha256sum crates/memra-server/src/worker.rs | cut -c1-16)"
  nice -n 10 cargo build --release -j 14 -p memra-server --bin memra-server > "$D/logs/build-$ARM.log" 2>&1
  rc=$?
  grep -E "Compiling memra-server|Finished" "$D/logs/build-$ARM.log" | tail -n 3
  if [ $rc = 0 ]; then
    cp -a "$CARGO_TARGET_DIR/release/memra-server" "$D/bin/memra-server-$ARM-120a"
    sha256sum "$D/bin/memra-server-$ARM-120a"
    strings "$D/bin/memra-server-$ARM-120a" | grep -o -m1 'memra-0\.[0-9]*\.[0-9]*-[0-9a-f]\{12\}'
  else
    echo "BUILD-FAILED $ARM"; grep -B2 -A10 '^error' "$D/logs/build-$ARM.log" | head -40
  fi
done
echo "BUILD-DONE $(date -Is)"
