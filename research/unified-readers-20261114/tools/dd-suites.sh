#!/usr/bin/env bash
# Dev desk, CPU only (memra#1307): the engine and server library suites on the head tree, after
# dd-checks.sh. CPUs 14-31.
set -u
D=$HOME/.cache/memra-1307
echo "$$" >> "$D/pids.txt"
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
L=$D/logs/${1:-suites}
mkdir -p "$L"
S=$L/summary.txt
cd "$D/tree" || exit 1
(cd crates/memra-engine/src && sha256sum spec.rs unified_memory.rs lib.rs dsv4_gpu.rs) > "$S"
taskset -c 14-31 nice -n 15 cargo test --release -p memra-engine --lib -j 14 > "$L/engine.log" 2>&1
echo "engine lib suite rc=$? $(grep -E '^test result' "$L/engine.log" | tail -n 1)" | tee -a "$S"
grep -E "^test .*FAILED$" "$L/engine.log" | head -n 10 >> "$S"
taskset -c 14-31 nice -n 15 cargo test --release -p memra-server --lib -j 14 > "$L/server.log" 2>&1
echo "server lib suite rc=$? $(grep -E '^test result' "$L/server.log" | tail -n 1)" | tee -a "$S"
grep -E "^test .*FAILED$" "$L/server.log" | head -n 10 >> "$S"
echo "SUITES-DONE $(date -Is)" | tee -a "$S"
