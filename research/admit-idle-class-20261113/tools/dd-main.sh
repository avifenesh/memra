#!/usr/bin/env bash
# Dev desk: after the fold's chain (pid given) has ended, memra-server (sm_120a, release) of MAIN
# for the RTX 5090 publisher cell: the four crate files the branch changes are swapped for
# main's (main-files/), the binary is built into bins/memra-server-main, and the tree's files are
# put back from the kept copies. CPUs 14-31.
# usage: dd-main.sh CHAIN_PID  -> bins/memra-server-main, logs/fold/main-build.done
set -u
D=$HOME/.cache/memra-1304
tail --pid="$1" -f /dev/null
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-check
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
L=$D/logs/fold
FILES="crates/memra-server/src/worker.rs crates/memra-server/src/worker/host_glm.rs crates/memra-engine/src/dflash.rs crates/memra-engine/src/glm_spec.rs"
cd "$D/tree" || exit 1
mkdir -p "$D/head-keep"
for f in $FILES; do mkdir -p "$D/head-keep/$(dirname "$f")"; cp "$f" "$D/head-keep/$f"; cp "$D/main-files/$f" "$f"; done
{ echo "start $(date -Is)"; sha256sum $FILES | sed "s|$HOME|~|"; } > "$L/main-build.log"
taskset -c 14-31 nice -n 15 cargo build --release --locked -p memra-server --bin memra-server -j 14 >> "$L/main-build.log" 2>&1
echo "build rc=$?" >> "$L/main-build.log"
cp -a "$CARGO_TARGET_DIR/release/memra-server" "$D/bins/memra-server-main" && sha256sum "$D/bins/memra-server-main" | sed "s|$HOME|~|" >> "$L/main-build.log"
for f in $FILES; do cp "$D/head-keep/$f" "$f"; done
sha256sum $FILES | sed "s|$HOME|~|" >> "$L/main-build.log"
touch "$L/main-build.done"
