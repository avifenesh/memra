#!/usr/bin/env bash
# WP-A day 13 CPU gate battery. Every command runs under the same quota scope; each log is the
# command's full output; the .exit file is its exit status. No GPU cell here (the target-card
# cells are pro-single-day13/). Usage: run-gates.sh [rust|docs|all]
set -u
cd "$(git rev-parse --show-toplevel)"
out=research/spill-a-20260919/day13/gates
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
  run shellcheck-cells bash -n research/spill-a-20260919/pro-single-day13/build.sh research/spill-a-20260919/pro-single-day13/build2.sh research/spill-a-20260919/pro-single-day13/run-cell.sh research/spill-a-20260919/pro-single-day13/pinned-cell.sh research/spill-a-20260919/pro-single-day13/gate-cell.sh research/spill-a-20260919/pro-single-day13/gputest-cell.sh research/spill-a-20260919/pro-single-day13/driver.sh research/spill-a-20260919/pro-single-day13/driver2.sh research/spill-a-20260919/pro-single-day13/probe-cell.sh research/spill-a-20260919/pro-single-day13/driver3.sh
  run py-compile-replay python3 -m py_compile research/spill-a-20260919/wc-ab.py research/spill-a-20260919/pinned-read-probe.py
fi
