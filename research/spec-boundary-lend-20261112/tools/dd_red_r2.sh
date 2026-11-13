#!/usr/bin/env bash
# CPU build host (CPU build host), memra#1221 part 2 review round 1: each CPU red arm is the review head
# with one fix disarmed (box/red_patch.py); its test must FAIL there (it passes at the head, in the
# full chain). One tree (src-red, a copy of src) and one target dir, an arm at a time.
# usage: dd_red_r1.sh   -> red-r2.txt (one verdict line per arm), red-r2-ARM.log
set -u
R=$HOME/.cache/memra-1221b
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target-red CARGO_BUILD_JOBS=12
waitfor --grep 'FULL-DONE' "$R/full.txt" --timeout 120m --quiet || exit 1
OUT=$R/red-r2.txt
echo "# $(date -Is) review round 2 red arms (each must FAIL)" > "$OUT"
for arm in ledger-double:memra-server:loan_books_move_a_charge_once_and_always_to_a_reader \
           ledger-holder:memra-server:loan_books_move_a_charge_once_and_always_to_a_reader \
           evictable:memra-server:loan_books_move_a_charge_once_and_always_to_a_reader \
           preflight-lease:memra-server:loan_books_move_a_charge_once_and_always_to_a_reader \
           hold-budget:memra-server:loan_books_move_a_charge_once_and_always_to_a_reader \
           compact-retry:memra-server:a_failed_compaction_waits_for_the_bytes_it_needed \
           rewind-retry:memra-server:a_lend_refused_rewind_survives_an_out_of_memory_fresh_cache \
           restore-floor:memra-engine:checkpoint_rewinds_check_the_lends_before_writing \
           publish-copy:memra-server:spec_boundary_publish_lends_on_the_mtp_path; do
  name=${arm%%:*}; rest=${arm#*:}; pkg=${rest%%:*}; test=${rest#*:}
  rsync -a --delete "$R/src/" "$R/src-red/"
  python3 "$R/red_patch.py" "$R/src-red" "$name" >> "$OUT" 2>&1 || { echo "$name PATCH-FAILED" >> "$OUT"; continue; }
  (cd "$R/src-red" && taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo test --release --locked -p "$pkg" --lib -- "$test" > "$R/red-r2-$name.log" 2>&1)
  rc=$?
  if grep -q "test result: FAILED" "$R/red-r2-$name.log"; then verdict="FAILS (as it must)"
  elif [ $rc = 0 ]; then verdict="PASSES (the arm is not caught)"
  else verdict="DID NOT RUN (build error)"; fi
  echo "$name $pkg::$test rc=$rc $verdict | $(grep -m1 -E "panicked at|assertion" "$R/red-r2-$name.log" | cut -c1-200)" >> "$OUT"
done
echo RED-R2-DONE >> "$OUT"
