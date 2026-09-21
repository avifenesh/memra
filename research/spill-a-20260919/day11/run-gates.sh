#!/usr/bin/env bash
# WP-A day 11 CPU gate battery. Every command runs under the same quota scope; each log is the
# command's full output; the .exit file is its exit status. No GPU cell.
set -u
cd "$(git rev-parse --show-toplevel)"
out=research/spill-a-20260919/day11/gates
mkdir -p "$out"
run() {
  local name=$1; shift
  systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@" > "$out/$name.log" 2>&1
  echo $? > "$out/$name.exit"
  printf '%s exit=%s\n' "$name" "$(cat "$out/$name.exit")"
}
run fmt cargo fmt --all -- --check
run test-tier-kv cargo test -p memra-tier -p memra-kv --offline
run clippy env DOCS_RS=1 cargo clippy -p memra-engine -p memra-tier -p memra-kv --offline --all-targets -- -D warnings
run check-flags bash tools/check-flags.sh
run diff-check git diff --check
