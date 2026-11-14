#!/usr/bin/env bash
# Dev desk, CPU only (memra#1316): the checks of the change, on the tree ~/.cache/memra-1316/tree.
#   green    memra-runtime's tests (the retry policy's and its wiring's are in ctx_boot)
#   red arms the same tests on each mutation of mutants.py (budget-zero is the file the on-device
#            arm B is built from): every one must fail
#   fmt, clippy (runtime, engine and server, all targets), then the engine and server library suites
# cargo decides by mtime, and both the copy from the workstation and a red arm's restore can leave
# ctx_boot.rs older than the last build: the source is touched before each arm and the log must
# show cargo rebuilding memra-runtime.
# Every cargo pinned to CPUs 14-31 (memra's share of the dev desk). Logs and summary.txt go to
# ~/.cache/memra-1316/logs/LABEL/.
# usage: dd_checks.sh LABEL [quick]      (quick: green and red arms only)
set -u
D=$HOME/.cache/memra-1316
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
L=$D/logs/${1:-checks}
mkdir -p "$L"
S=$L/summary.txt
T=$D/tree
F=$T/crates/memra-runtime/src/ctx_boot.rs
cd "$T" || exit 1
run() { taskset -c 14-31 nice -n 15 "$@"; }
built() { grep -q 'Compiling memra-runtime' "$1" && echo rebuilt || echo NOT-REBUILT; }
sha256sum crates/memra-runtime/src/ctx_boot.rs crates/memra-runtime/src/lib.rs crates/memra-engine/src/lib.rs > "$S"
touch "$F"
run cargo test --release --locked -p memra-runtime -j 14 > "$L/green-runtime.log" 2>&1
echo "green memra-runtime rc=$? ($(built "$L/green-runtime.log")) $(grep -E '^test result' "$L/green-runtime.log" | head -n 1)" | tee -a "$S"
cp "$F" "$L/ctx_boot.keep"
: > "$L/red-arms.log"
for m in $(python3 "$D/mutants.py" list); do
  cp "$L/ctx_boot.keep" "$F"
  python3 "$D/mutants.py" "$m" "$F" > "$L/red-$m.log" 2>&1
  if cmp -s "$L/ctx_boot.keep" "$F"; then echo "red $m ARM-NOT-APPLIED" | tee -a "$S"; continue; fi
  touch "$F"
  run cargo test --release --locked -p memra-runtime --lib -j 14 -- ctx_boot >> "$L/red-$m.log" 2>&1
  rc=$?
  echo "red $m rc=$rc ($(built "$L/red-$m.log")) $(grep -E '^test result|^error' "$L/red-$m.log" | head -n 1) failing: $(grep -E '^test .*FAILED$' "$L/red-$m.log" | sed -E 's/^test ctx_boot::tests::(.*) \.\.\. FAILED$/\1/' | tr '\n' ' ')" | tee -a "$S"
  { echo "=== $m"; grep -E '^test |panicked at|^  left|^ right|^test result|^error' "$L/red-$m.log"; } >> "$L/red-arms.log"
done
cp "$L/ctx_boot.keep" "$F"
touch "$F"
[ "${2:-}" = quick ] && { echo "QUICK-DONE $(date -Is)" | tee -a "$S"; exit 0; }
run cargo fmt --all -- --check > "$L/fmt.log" 2>&1; echo "fmt rc=$?" | tee -a "$S"
run cargo clippy --release --locked -p memra-runtime -p memra-engine -p memra-server --all-targets -j 14 -- -D warnings > "$L/clippy.log" 2>&1
echo "clippy rc=$?" | tee -a "$S"
grep -E '^(error|warning)' "$L/clippy.log" | grep -v 'nvcc\|MEMRA_CUDA_ARCH' | sort | uniq -c | head -n 5 >> "$S"
run cargo test --release --locked -p memra-engine --lib -j 14 > "$L/engine.log" 2>&1
echo "engine lib suite rc=$? $(grep -E '^test result' "$L/engine.log" | tail -n 1)" | tee -a "$S"
grep -E '^test .*FAILED$' "$L/engine.log" | head -n 10 >> "$S"
run cargo test --release --locked -p memra-server --lib -j 14 > "$L/server.log" 2>&1
echo "server lib suite rc=$? $(grep -E '^test result' "$L/server.log" | tail -n 1)" | tee -a "$S"
grep -E '^test .*FAILED$' "$L/server.log" | head -n 10 >> "$S"
echo "CHECKS-DONE $(date -Is)" | tee -a "$S"
