#!/usr/bin/env bash
# Day 10 flake battery: N runs of the memra-tier storage test binary under a
# CPU/memory quota. Each run records the UTC start, the load line, the
# co-resident processes at start, the full test output and the exit code.
# Usage: run-battery.sh <label> <count> [extra test-binary args...]
set -u
label="$1"; count="$2"; shift 2
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/$label"; mkdir -p "$out"
cd "$(git -C "$here" rev-parse --show-toplevel)"
{
  echo "label=$label count=$count args=$*"
  echo "head=$(git rev-parse HEAD)"
  echo "tmpdir=${TMPDIR:-/tmp} fstype=$(stat -f -c %T "${TMPDIR:-/tmp}") kernel=$(uname -r)"
  echo "rustc=$(rustc --version)"
  echo "nproc=$(nproc)"
} > "$out/context.txt"
for i in $(seq 1 "$count"); do
  r="$out/run-$i"
  { date -u +%Y-%m-%dT%H:%M:%SZ; uptime; } > "$r.start"
  ps -eo pid,pcpu,pmem,etime,comm --sort=-pcpu | head -25 > "$r.procs"
  systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G \
    cargo test -p memra-tier --test storage --offline -- "$@" > "$r.log" 2>&1
  echo $? > "$r.exit"
  { echo "exit=$(cat "$r.exit")"; grep -E '^test result|panicked at|Err\(Busy\)|FAILED|failures:$' "$r.log" | head -20; } > "$r.summary"
done
for i in $(seq 1 "$count"); do echo "run-$i exit=$(cat "$out/run-$i.exit") $(grep -E '^test result' "$out/run-$i.log" | tail -1)"; done | tee "$out/RESULTS.txt"
