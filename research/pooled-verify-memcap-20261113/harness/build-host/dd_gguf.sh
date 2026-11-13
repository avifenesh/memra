#!/usr/bin/env bash
# CPU build host, memra#1266: rerun memra-gguf lib (release, as the full chain runs it) three times to
# test the hf_mapping temp-config race that failed one test in the round-2 chain.
set -u
R=<lane>
export PATH="$HOME/.cargo/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target
cd "$R/src" || exit 1
for i in 1 2 3; do
  taskset -c 20-31 nice -n 15 cargo test --release --locked -p memra-gguf --lib > "$R/gguf-rerun-$i.log" 2>&1
  echo "run $i rc=$? $(grep -E "^test result" "$R/gguf-rerun-$i.log" | head -n 1)"
done > "$R/gguf-rerun.txt"
echo GGUF-DONE >> "$R/gguf-rerun.txt"
