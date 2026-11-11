#!/usr/bin/env bash
# Runs on a CPU build host (CUDA 13.1 compiler, sm_120a build, no GPU): the branch's CPU checks at
# one commit, from a source tree copied with git archive.
#   fmt, workspace clippy -D warnings on all targets, the memra-server and memra-engine suites, then the census on
#   the fail-before tree (base production code plus the head's tests; tools/red_worker.py).
# usage: cpu_checks.sh SRC_DIR SHA RED_WORKER   -> $R/logs/cpu-checks.txt, last line CPU-CHECKS-DONE
set -u
SRC=$1; SHA=$2; RED=$3
R=$HOME/.cache/memra-specpub
L=$R/logs; mkdir -p "$L"
O=$L/cpu-checks.txt
export PATH="$HOME/.cargo/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target
cd "$SRC" || exit 1
W=crates/memra-server/src/worker.rs
T=spec_boundary_publish_allocates_no_zeroed_plane
{
  echo "# CPU checks at $SHA (sm_120a build, cargo --release --locked)"
  echo "# cargo fmt --all -- --check"
  cargo fmt --all -- --check > "$L/fmt.log" 2>&1; echo "fmt $SHA rc=$?"
  echo "# workspace clippy --all-targets -D warnings"
  nice -n 10 cargo clippy --release --locked --workspace --all-targets -- -D warnings > "$L/clippy.log" 2>&1
  echo "clippy $SHA rc=$?"; grep -E '^(warning|error)' "$L/clippy.log" | sort | uniq -c | head -20; tail -n 1 "$L/clippy.log"
  echo "# the memra-server suite"
  nice -n 10 cargo test --release --locked -p memra-server > "$L/server-test.log" 2>&1; rc=$?
  grep -E '^test result|FAILED|panicked' "$L/server-test.log" | head -40
  echo "server $SHA rc=$rc"
  echo "# the memra-engine suite"
  nice -n 10 cargo test --release --locked -p memra-engine > "$L/engine-test.log" 2>&1; rc=$?
  grep -E '^test result|FAILED|panicked' "$L/engine-test.log" | head -40
  echo "engine $SHA rc=$rc"
  echo "# the census at the head"
  grep -E "^test .*$T" "$L/server-test.log"
  echo "# the census on the fail-before tree (the base's production code, the head's tests)"
  cp "$W" "$L/worker.head.rs"; cp "$RED" "$W"
  nice -n 10 cargo test --release --locked -p memra-server --lib "$T" > "$L/red-test.log" 2>&1; rc=$?
  grep -E "^test .*$T|^test result|the spec publisher calls" "$L/red-test.log" | head -10
  echo "red rc=$rc"
  cp "$L/worker.head.rs" "$W"; touch "$W"
  echo CPU-CHECKS-DONE
} > "$O" 2>&1
