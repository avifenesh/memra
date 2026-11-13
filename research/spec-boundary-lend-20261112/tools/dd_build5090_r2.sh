#!/usr/bin/env bash
# CPU build host: the RTX 5090 (sm_120a) review-round-2 binaries from src, after the round-2 red arms:
# the server lib test binary and the session gate. -> bins5090/R2/{server-lib-test,session-gate}
set -u
R=$HOME/.cache/memra-1221b
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target
L=$R/build5090-r2.log; : > "$L"
waitfor --grep 'RED-R2-DONE' "$R/red-r2.txt" --timeout 180m --quiet || { echo "BUILD-FAIL (red arms not done)" >> "$L"; exit 1; }
cd "$R/src" || exit 1
mkdir -p "$R/bins5090/R2"
taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo test --release --locked -j 12 -p memra-server --lib --no-run --message-format=json > "$R/tests5090-r2.json" 2>> "$L" || { echo BUILD-FAIL tests >> "$L"; exit 1; }
exe=$(grep -o '"executable":"[^"]*"' "$R/tests5090-r2.json" | grep -v null | grep memra_server | tail -1 | cut -d'"' -f4)
cp -a "$exe" "$R/bins5090/R2/server-lib-test"
taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo build --release --locked -j 12 -p memra-engine --bin session-gate >> "$L" 2>&1 || { echo BUILD-FAIL gate >> "$L"; exit 1; }
cp -a "$CARGO_TARGET_DIR/release/session-gate" "$R/bins5090/R2/"
sha256sum "$R/bins5090/R2/"* >> "$L"
echo BUILD-DONE >> "$L"
