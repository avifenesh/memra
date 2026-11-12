#!/usr/bin/env bash
# Dev desk (memra#1173): CPU checks of the synced fix tree. usage: dd-checks.sh [check|tests FILTER|full]
set -u
D=$HOME/.cache/memra-1173
mkdir -p "$D/logs"
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
echo "$$" >> "$D/pids.txt"
cd "$D/tree" || exit 1
step() {
  local name=$1; shift
  echo "== $name: $(date -Is)"
  nice -n 15 ionice -c2 -n7 "$@" > "$D/logs/check-$name.log" 2>&1
  local rc=$?
  echo "== $name rc=$rc"
  grep -E "^test result|^error|^warning|FAILED|panicked" "$D/logs/check-$name.log" | sort | uniq -c | sort -rn | head -n 12
  return $rc
}
fail=0
if [ "${1:-}" = check ]; then
  step check cargo check --release -p memra-server --all-targets -j 14 || fail=1
  grep -E -A14 '^(error|warning)' "$D/logs/check-check.log" | head -150
  echo "CHECKS-DONE fail=$fail $(date -Is)"; exit $fail
fi
if [ "${1:-}" = tests ]; then
  step tests cargo test --release -p memra-server -j 14 -- "${2:-drafter}" "${3:-vram_gate}" || fail=1
  grep -E -B2 -A25 '^(error|failures:)|panicked' "$D/logs/check-tests.log" | head -150
  echo "CHECKS-DONE fail=$fail $(date -Is)"; exit $fail
fi
step fmt cargo fmt --all -- --check || fail=1
step clippy cargo clippy --release --all-targets -j 14 -- -D warnings || fail=1
step test-server cargo test --release -p memra-server -j 14 || fail=1
step flags bash tools/check-flags.sh || fail=1
step numeric python3 tools/check-numeric-scope.py || fail=1
echo "CHECKS-DONE fail=$fail $(date -Is)"
