#!/usr/bin/env bash
# Build host, memra#1266: one red arm again (tools/red_arms.py), as dd_red.sh runs it.
# usage: dd_red_one.sh ARM CRATE FILTER   -> <lane>/red2/ARM.log, summary.txt row
set -u
R=<lane>; O=$R/red2
export PATH="$HOME/.cargo/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target-test
python3 "$R/red_arms.py" "$R/src" "$O" "$1" > "$O/$1.log" 2>&1 || { echo "$1 PATCH-FAIL" >> "$O/summary.txt"; exit 1; }
( cd "$R/src" && taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo test --locked -j 8 -p "$2" --lib "$3" ) >> "$O/$1.log" 2>&1
rc=$?
echo "$1 rc=$rc $(grep -m1 -E "panicked at|error\[" "$O/$1.log" | cut -c1-200) (rerun)" >> "$O/summary.txt"
python3 "$R/red_arms.py" "$R/src" "$O" restore >> "$O/$1.log" 2>&1
echo "ONE-DONE $1" >> "$O/summary.txt"
