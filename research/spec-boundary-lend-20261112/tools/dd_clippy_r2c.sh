#!/usr/bin/env bash
# CPU build host: after the 5090 R2c build, fmt and clippy -D warnings (all targets) on memra-server at
# the R2c tree (src, 4947505535: R2 plus a test-only room guard in one GPU cell). -> clippy-r2c.txt
set -u
R=$HOME/.cache/memra-1221b
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target
O=$R/clippy-r2c.txt
waitfor --grep 'BUILD-DONE|BUILD-FAIL' "$R/build5090-r2.log" --timeout 120m --quiet || { echo "CLIPPY-SKIP (build not done)" > "$O"; exit 1; }
cd "$R/src" || exit 1
{ echo "# $(date -Is) R2c $(cat .head) fmt + clippy memra-server all targets"
  cargo fmt --all -- --check > "$R/clippy-r2c-fmt.log" 2>&1; echo "fmt rc=$?"
  taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo clippy --release --locked -j 12 -p memra-server --all-targets -- -D warnings > "$R/clippy-r2c.log" 2>&1; echo "clippy rc=$?"
  tail -n 1 "$R/clippy-r2c.log"
  echo CLIPPY-DONE; } > "$O" 2>&1
