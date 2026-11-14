#!/usr/bin/env bash
# Dev desk: after the fold's chain (dd-chain.sh, pid given) has ended, the whole memra-engine
# library suite on the same tree and target dir (dflash.rs and glm_spec.rs changed). CPUs 14-31.
# usage: dd-post.sh CHAIN_PID LABEL  -> logs/fold/LABEL/engine-lib.log, logs/fold/LABEL.done
set -u
D=$HOME/.cache/memra-1304
tail --pid="$1" -f /dev/null
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-check
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
L=$D/logs/fold/$2
mkdir -p "$L"
cd "$D/tree" || exit 1
{ echo "start $(date -Is)"; sha256sum crates/memra-server/src/worker.rs crates/memra-engine/src/dflash.rs crates/memra-engine/src/glm_spec.rs | sed "s|$HOME|~|"; } > "$L/summary.txt"
taskset -c 14-31 nice -n 15 cargo test --release -p memra-engine --lib -j 14 > "$L/engine-lib.log" 2>&1
echo "engine lib suite rc=$? $(grep -E '^test result' "$L/engine-lib.log" | tail -n 1)" >> "$L/summary.txt"
grep -E "^test .*FAILED$" "$L/engine-lib.log" | head -n 20 >> "$L/summary.txt"
echo "done $(date -Is)" >> "$L/summary.txt"
touch "$D/logs/fold/$2.done"
