#!/usr/bin/env bash
# Rig: sync the worktree's code to the CPU build host tree and run one cargo command there at low
# weight (dd_check.sh), waiting for it. usage: dd_sync_check.sh NAME CARGO_ARGS...
# -> prints the rc and the log's errors; full log at build-host:~/.cache/memra-1221b/NAME.log
set -u
NAME=$1; shift
cd "$HOME/.cache/memra-1221b/wt" || exit 1
rsync -a --delete --exclude=target crates docs tools probe deploy Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml build-host:.cache/memra-1221b/src/ || exit 1
ssh build-host "(setsid nohup bash ~/.cache/memra-1221b/dd_check.sh $NAME $*) > /dev/null 2>&1 < /dev/null & echo started"
waitfor --ssh build-host "test -f ~/.cache/memra-1221b/$NAME.rc" --timeout "${WAIT:-40m}" --every 15s --quiet
ssh build-host "cat ~/.cache/memra-1221b/$NAME.rc; grep -E -A12 '^(error|warning: unused)' ~/.cache/memra-1221b/$NAME.log | head -${LINES_MAX:-150}; tail -2 ~/.cache/memra-1221b/$NAME.log"
