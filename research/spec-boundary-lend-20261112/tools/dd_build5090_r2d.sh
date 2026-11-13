#!/usr/bin/env bash
# CPU build host: the RTX 5090 (sm_120a) binaries of R2d (842a6b350c: R2c plus the numeric scope at
# spec_rewind_into_fresh's entry) from src: the server lib test binary and the session gate.
# -> bins5090/R2d/{server-lib-test,session-gate}, build5090-r2d.log (last line BUILD-DONE)
set -u
R=$HOME/.cache/memra-1221b
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target
L=$R/build5090-r2d.log; : > "$L"
cd "$R/src" || exit 1
mkdir -p "$R/bins5090/R2d"
taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo test --release --locked -j 12 -p memra-server --lib --no-run --message-format=json > "$R/tests5090-r2d.json" 2>> "$L" || { echo BUILD-FAIL tests >> "$L"; exit 1; }
exe=$(grep -o '"executable":"[^"]*"' "$R/tests5090-r2d.json" | grep -v null | grep memra_server | tail -1 | cut -d'"' -f4)
cp -a "$exe" "$R/bins5090/R2d/server-lib-test"
taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo build --release --locked -j 12 -p memra-engine --bin session-gate >> "$L" 2>&1 || { echo BUILD-FAIL gate >> "$L"; exit 1; }
cp -a "$CARGO_TARGET_DIR/release/session-gate" "$R/bins5090/R2d/"
taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo clippy --release --locked -j 12 -p memra-engine -p memra-server --all-targets -- -D warnings > "$R/clippy-r2d.log" 2>&1; echo "clippy rc=$?" >> "$L"
sha256sum "$R/bins5090/R2d/"* >> "$L"
echo BUILD-DONE >> "$L"
