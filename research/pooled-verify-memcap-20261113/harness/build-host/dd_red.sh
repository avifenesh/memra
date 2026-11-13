#!/usr/bin/env bash
# Build host, memra#1266: the CPU red arms (tools/red_arms.py), one after another on the synced
# tree, each test run at low weight on cores 20-31 in target-test; the fix is kept first and
# restored after each arm. -> $RED_OUT (default <lane>/red2)/{ARM.log,summary.txt}
set -u
R=<lane>; O=${RED_OUT:-$R/red2}; mkdir -p "$O"
export PATH="$HOME/.cargo/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target-test
: > "$O/summary.txt"
python3 "$R/red_arms.py" "$R/src" "$O" keep > "$O/keep.log" 2>&1 || { echo "KEEP-FAIL" >> "$O/summary.txt"; exit 1; }
run() { # ARM CRATE FILTER
  python3 "$R/red_arms.py" "$R/src" "$O" "$1" > "$O/$1.log" 2>&1 || { echo "$1 PATCH-FAIL" >> "$O/summary.txt"; python3 "$R/red_arms.py" "$R/src" "$O" restore >> "$O/$1.log" 2>&1; return; }
  ( cd "$R/src" && taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo test --locked -j 8 -p "$2" --lib "$3" ) >> "$O/$1.log" 2>&1
  rc=$?
  echo "$1 rc=$rc $(grep -m1 -E "panicked at|error\[" "$O/$1.log" | cut -c1-200)" >> "$O/summary.txt"
  python3 "$R/red_arms.py" "$R/src" "$O" restore >> "$O/$1.log" 2>&1
}
run accumulation-off memra-engine pooled_verify_cap_1266
run fa-growth-off memra-engine pooled_verify_cap_1266
run rider-cap-off memra-engine pooled_verify_cap_1266
run margin-low memra-engine pooled_verify_cap_1266
run main-engine memra-engine pooled_verify_cap_1266
run snapshots-off memra-server pre_cohort_reclaim_1266
run parked-first memra-server pre_cohort_reclaim_1266
run unreadable-acts memra-server pre_cohort_reclaim_1266
run main-worker memra-server pre_cohort_reclaim_1266
echo "RED-DONE $(date -Is)" >> "$O/summary.txt"
