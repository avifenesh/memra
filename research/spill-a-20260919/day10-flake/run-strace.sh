#!/usr/bin/env bash
# Capture the syscall origin of a Busy: run the storage test binary under
# strace (flock, clone/clone3/vfork, execve, close, exit_group; -y resolves fds
# to paths) at default threads, up to N attempts, stop at the first failing run.
# Passing attempts keep their test log and exit code; their strace is gzipped.
set -u
count="$1"
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/strace"; mkdir -p "$out"
cd "$(git -C "$here" rev-parse --show-toplevel)"
bin="$(cargo test -p memra-tier --test storage --offline --no-run 2>&1 | sed -n 's/.*(\(target\/debug\/deps\/storage-[0-9a-f]*\)).*/\1/p' | head -1)"
echo "binary=$bin sha256=$(sha256sum "$bin" | cut -c1-16) head=$(git rev-parse HEAD)" > "$out/context.txt"
for i in $(seq 1 "$count"); do
  a="$out/attempt-$i"
  { date -u +%Y-%m-%dT%H:%M:%SZ; uptime; } > "$a.start"
  RUST_BACKTRACE=1 systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G \
    strace -f -tt -y -qq -e signal=none -e trace=flock,clone,clone3,vfork,execve,close,exit_group \
    -o "$a.strace" "$bin" --nocapture > "$a.log" 2>&1
  echo $? > "$a.exit"
  echo "attempt-$i exit=$(cat "$a.exit") $(grep -E '^test result' "$a.log" | tail -1)" | tee -a "$out/RESULTS.txt"
  if grep -q 'value: Busy' "$a.log"; then echo "captured Busy at attempt $i" | tee -a "$out/RESULTS.txt"; break; fi
  gzip -f "$a.strace"
done
