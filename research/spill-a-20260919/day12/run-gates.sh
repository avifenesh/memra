#!/usr/bin/env bash
# WP-A day 12 CPU gate battery. Every command runs under the same quota scope; each log is the
# command's full output; the .exit file is its exit status. No GPU cell here (the target-card
# cells are pro-single-day12/). Usage: run-gates.sh [rust|docs|all]
set -u
cd "$(git rev-parse --show-toplevel)"
out=research/spill-a-20260919/day12/gates
mkdir -p "$out"
run() {
  local name=$1; shift
  systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@" > "$out/$name.log" 2>&1
  echo $? > "$out/$name.exit"
  printf '%s exit=%s\n' "$name" "$(cat "$out/$name.exit")"
}
which=${1:-all}
if [ "$which" = rust ] || [ "$which" = all ]; then
  run fmt cargo fmt --all -- --check
  run test-server cargo test -p memra-server --offline
  run test-kv cargo test -p memra-kv --offline
  run clippy-server cargo clippy -p memra-server --offline --all-targets -- -D warnings
  run clippy-kv cargo clippy -p memra-kv --offline --all-targets -- -D warnings
fi
if [ "$which" = docs ] || [ "$which" = all ]; then
  run check-flags bash tools/check-flags.sh
  run docs-registry-census bash tools/docs-registry-census.sh
  run diff-check git diff --check
  run shellcheck-gate bash -n tools/kv-host-tenant-reclaim-gate.sh
  run pyflakes-harness python3 -m py_compile tools/pinned-host-reserve-bench.py
fi
