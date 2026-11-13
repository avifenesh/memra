#!/usr/bin/env bash
# CPU build host (CPU build host), memra#1221 part 2 review round 3: each CPU red arm is the round-3 head
# with one fix disarmed (box/red_patch.py); its test must FAIL there (it passes at the head, in the
# full chain). The two new arms (budget-tick, hold-stuck) and the two round-2 arms whose code the
# round changed (hold-budget, rewind-retry). One tree (src-red, a copy of src) and one target dir,
# an arm at a time, after the full chain.
# usage: dd_red_r3.sh   -> red-r3.txt (one verdict line per arm), red-r3-ARM.log
set -u
R=$HOME/.cache/memra-1221b
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target-red CARGO_BUILD_JOBS=12
waitfor --grep 'FULL-DONE' "$R/full.txt" --timeout 120m --quiet || exit 1
OUT=$R/red-r3.txt
echo "# $(date -Is) review round 3 red arms at $(cat "$R/src/.head") (each must FAIL)" > "$OUT"
for arm in budget-tick:memra-server:the_tick_and_retirement_reconciles_hold_the_budget \
           hold-stuck:memra-server:loan_books_move_a_charge_once_and_always_to_a_reader \
           hold-budget:memra-server:loan_books_move_a_charge_once_and_always_to_a_reader \
           rewind-retry:memra-server:a_lend_refused_rewind_survives_an_out_of_memory_fresh_cache; do
  name=${arm%%:*}; rest=${arm#*:}; pkg=${rest%%:*}; test=${rest#*:}
  rsync -a --delete "$R/src/" "$R/src-red/"
  python3 "$R/red_patch.py" "$R/src-red" "$name" >> "$OUT" 2>&1 || { echo "$name PATCH-FAILED" >> "$OUT"; continue; }
  (cd "$R/src-red" && taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo test --release --locked -p "$pkg" --lib -- "$test" > "$R/red-r3-$name.log" 2>&1)
  rc=$?
  if grep -q "test result: FAILED" "$R/red-r3-$name.log"; then verdict="FAILS (as it must)"
  elif [ $rc = 0 ]; then verdict="PASSES (the arm is not caught)"
  else verdict="DID NOT RUN (build error)"; fi
  echo "$name $pkg::$test rc=$rc $verdict | $(grep -m1 -A1 -E "panicked at" "$R/red-r3-$name.log" | tail -n 1 | cut -c1-200)" >> "$OUT"
done
echo RED-R3-DONE >> "$OUT"
