#!/usr/bin/env bash
# Runs ON the dev desk (memra#1227). Red arms, each of which one of the three tests must fail:
#   main      the base's worker.rs with only the two pure functions and the tests added (phase (a)
#             as before the change): the wiring test fails.
#   oldorder  spec_cold_serial_first names nobody (the cohorts step first): the naming test fails.
#   nojoin    a session that primed first is never handled in the cohort pass: the pass test fails.
#   swapped   the cohort pass runs first: the wiring test fails.
#   nopeer    every newcomer is named, trimmed prime or not (a lone speculative newcomer beside
#             plain riders then bursts twice in one tick): the wiring test fails.
#   noquantum the cohort pass does not read the prime quantum before the cohorts form (a cohort
#             then runs a whole burst between a long prime's chunks): the wiring test fails.
# Then green: all three pass.
set -u
D=$HOME/.cache/memra-1227
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
cd "$D/tree" || exit 1
W=crates/memra-server/src/worker.rs
cp "$W" "$D/worker.green.rs"
TESTS="spec_cold_serial_first_names_the_newcomers spec_pass_steps_each_step_once spec_newcomers_prime_before_cohorts_is_wired"
run() {
  local name=$1
  echo "== $name: $(date -Is)"
  for t in $TESTS; do
    nice -n 15 cargo test --release -p memra-server --lib -j 14 "$t" >> "$D/redgreen-$name.log" 2>&1
    echo "$name $t rc=$?"
  done
  grep -E "^test .*(ok|FAILED)$|panicked at|test result|^error" "$D/redgreen-$name.log" | head -n 30
}
rm -f "$D"/redgreen-*.log
for arm in main oldorder nojoin swapped nopeer noquantum; do
  python3 "$D/tools/redarm.py" "$arm" "$D/worker.green.rs" "$D/worker.base.rs" "$W" || { echo "red arm $arm did not apply"; cp "$D/worker.green.rs" "$W"; exit 1; }
  run "red-$arm"
done
cp "$D/worker.green.rs" "$W"
run green
echo "REDGREEN-DONE $(date -Is)"
