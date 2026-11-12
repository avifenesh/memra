#!/usr/bin/env bash
# Runs ON the dev desk (memra#1244). Red arms (tools/redarm.py), each of which must fail the
# tests its docstring names, then green: all three pass.
set -u
D=$HOME/.cache/memra-1244
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
cd "$D/tree" || exit 1
W=crates/memra-server/src/worker.rs
cp "$W" "$D/worker.green.rs"
TESTS="spec_round_yield_due_names_the_plain_sessions_behind_the_bursts spec_burst_ends_at_its_first_round_while_a_plain_session_waits spec_round_yield_is_wired"
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
for arm in parent nohook nodue nowrite noride; do
  python3 "$D/tools/redarm.py" "$arm" "$D/worker.green.rs" "$W" || { echo "red arm $arm did not apply"; cp "$D/worker.green.rs" "$W"; exit 1; }
  run "red-$arm"
done
cp "$D/worker.green.rs" "$W"
run green
echo "REDGREEN-DONE $(date -Is)"
