#!/usr/bin/env bash
# CPU build host (CPU build host, no GPU), memra#1221 part 2: the RTX 5090 (sm_120a) servers of the
# agentic and no-regression cells. ST = the head tree (src, as synced) with arms' worker.ST.rs (head
# + the leak-check trim probe); BT = the same tree with main 11b368f7d7's four changed crate files
# and worker.BT.rs (main + the trace probe + the trim probe). Own target dirs, cores 20-31, low weight.
# -> ~/.cache/memra-1221b/bins5090/{ST,BT}/memra-server, build5090.log
set -u
R=$HOME/.cache/memra-1221b
export PATH="$HOME/.cargo/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
L=$R/build5090.log; : > "$L"
b() { # TREE TARGET OUT
  ( cd "$1" && CARGO_TARGET_DIR=$2 taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo build --release --locked -j 12 -p memra-server --bin memra-server ) >> "$L" 2>&1 || return 1
  mkdir -p "$R/bins5090/$3"; cp -a "$2/release/memra-server" "$R/bins5090/$3/"; sha256sum "$R/bins5090/$3/memra-server" >> "$L"
}
# ST: the head tree with the trim probe over worker.rs, put back afterwards.
W=$R/src/crates/memra-server/src/worker.rs
cp "$W" "$R/worker.head.keep.rs"; cp "$R/main-files/worker.ST.rs" "$W"
b "$R/src" "$R/target-st" ST; rc=$?
cp "$R/worker.head.keep.rs" "$W"
[ $rc = 0 ] || { echo BUILD-FAIL ST >> "$L"; exit 1; }
# BT: a hardlinked copy of the tree with main's files (written as new files, so the links break).
[ -d "$R/src-main" ] || cp -al "$R/src" "$R/src-main"
for f in crates/memra-kv/src/lib.rs crates/memra-kv/src/plane.rs crates/memra-engine/src/spec.rs crates/memra-server/src/worker/host_glm.rs; do
  rm -f "$R/src-main/$f"; cp "$R/main-files/$f" "$R/src-main/$f"
done
rm -f "$R/src-main/crates/memra-server/src/worker.rs"; cp "$R/main-files/worker.BT.rs" "$R/src-main/crates/memra-server/src/worker.rs"
b "$R/src-main" "$R/target-bt" BT || { echo BUILD-FAIL BT >> "$L"; exit 1; }
echo BUILD-DONE >> "$L"
