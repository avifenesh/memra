#!/usr/bin/env bash
# Day 10 CPU gate under the quota scope. Logs and exit codes land in gates/.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/gates"; mkdir -p "$out"
cd "$(git -C "$here" rev-parse --show-toplevel)"
q() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
run() { local name="$1"; shift; "$@" > "$out/$name.log" 2>&1; echo $? > "$out/$name.exit"; echo "$name exit=$(cat "$out/$name.exit")"; }
run fmt-check q cargo fmt --all -- --check
run test-tier-kv q cargo test -p memra-tier -p memra-kv --offline
run clippy-tier-kv q cargo clippy -p memra-tier -p memra-kv --offline --all-targets -- -D warnings
run clippy-engine-docsrs q env DOCS_RS=1 cargo clippy -p memra-engine --offline --all-targets -- -D warnings
run check-flags q bash tools/check-flags.sh
run diff-check git diff --check
