#!/usr/bin/env bash
# Dev desk: the red arms of memra#1304 and its two review folds, tests kept. The arms run in the
# batches mutate.py names (no two arms of a batch name the same test, none breaks another's): for
# each batch, apply its arms to the tree's worker.rs, build the test binary once, run the test
# each arm names, require that it FAILS and that its output contains the arm's needle (the stated
# reason), restore the file. Ends by checking the file is byte-identical to the kept copy.
# Every cargo pinned to CPUs 14-31.
# usage: dd-red.sh LABEL [ARM...]   (default: every batch; with ARMs, those arms as one batch)
#        -> logs/fold/LABEL/, logs/fold/LABEL.done
set -u
D=$HOME/.cache/memra-1304
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-check
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
LABEL=$1; shift
L=$D/logs/fold/$LABEL
mkdir -p "$L"
rm -f "$D/logs/fold/$LABEL.done"
S=$L/summary.txt
W=$D/tree/crates/memra-server/src/worker.rs
M=$D/mutate.py
cd "$D/tree" || exit 1
cp "$W" "$L/worker.keep"
{ echo "start $(date -Is)"; sha256sum "$W" | sed "s|$HOME|~|"; } > "$S"
if [ $# -gt 0 ]; then echo "$*" > "$L/batches.txt"; else python3 "$M" --batches > "$L/batches.txt" || exit 1; fi
n=0
while read -r batch; do
  n=$((n+1))
  cp "$L/worker.keep" "$W"
  echo "batch $n: $batch" >> "$S"
  # shellcheck disable=SC2086
  python3 "$M" "$W" $batch >> "$S" 2>&1 || { echo "batch $n MUTATION-FAILED" >> "$S"; continue; }
  taskset -c 14-31 nice -n 15 cargo test --release -p memra-server --lib -j 14 --no-run > "$L/batch$n-build.log" 2>&1
  echo "batch $n build rc=$?" >> "$S"
  for arm in $batch; do
    row=$(python3 "$M" --list | awk -F'\t' -v a="$arm" '$1==a')
    test=$(printf '%s' "$row" | cut -f2); needle=$(printf '%s' "$row" | cut -f3)
    taskset -c 14-31 nice -n 15 cargo test --release -p memra-server --lib -j 14 -- --exact "worker::tests::$test" > "$L/$arm-$test.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -qE '^test result: FAILED\. 0 passed; 1 failed' "$L/$arm-$test.log" && grep -qF -- "$needle" "$L/$arm-$test.log"; then v=RED-AS-STATED; else v=NOT-RED; fi
    echo "$arm $v rc=$rc $test $(grep -E '^test result' "$L/$arm-$test.log" | tail -n 1)" >> "$S"
    grep -A5 "panicked at" "$L/$arm-$test.log" | head -n 7 | cut -c1-300 >> "$S"
  done
done < "$L/batches.txt"
cp "$L/worker.keep" "$W"
cmp "$W" "$L/worker.keep" && echo "worker.rs restored" >> "$S"
sha256sum "$W" | sed "s|$HOME|~|" >> "$S"
echo "done $(date -Is)" >> "$S"
touch "$D/logs/fold/$LABEL.done"
