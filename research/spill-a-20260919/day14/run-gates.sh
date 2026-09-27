#!/usr/bin/env bash
# WP-A day 14 CPU gate battery. Every command runs under the same quota scope; each log is the
# command's full output; the .exit file is its exit status. No GPU cell here (the card
# cells are rtx5090-day14/ and pro-single-day14/). Usage: run-gates.sh [rust|docs|all]
set -u
cd "$(git rev-parse --show-toplevel)"
out=research/spill-a-20260919/day14/gates
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
  run test-lib cargo test -p memra-tier -p memra-kv -p memra-engine --offline --lib
  run clippy cargo clippy -p memra-engine -p memra-tier -p memra-kv --offline --all-targets -- -D warnings
fi
if [ "$which" = docs ] || [ "$which" = all ]; then
  run check-flags bash tools/check-flags.sh
  run docs-registry-census bash tools/docs-registry-census.sh
  run diff-check git diff --check
  run shellcheck-cells bash -n research/spill-a-20260919/pro-single-day14/build.sh research/spill-a-20260919/pro-single-day14/run-cell.sh research/spill-a-20260919/pro-single-day14/gate-cell.sh research/spill-a-20260919/pro-single-day14/gputest-cell.sh research/spill-a-20260919/pro-single-day14/driver.sh research/spill-a-20260919/rtx5090-day14/run-cell.sh research/spill-a-20260919/rtx5090-day14/pinned-cell.sh research/spill-a-20260919/rtx5090-day14/gate-cell.sh research/spill-a-20260919/rtx5090-day14/gputest-cell.sh research/spill-a-20260919/rtx5090-day14/driver.sh research/spill-a-20260919/rtx5090-day14/driver2.sh
  run py-compile-replay python3 -m py_compile research/spill-a-20260919/wc-ab.py research/spill-a-20260919/pinned-read-probe.py
fi
