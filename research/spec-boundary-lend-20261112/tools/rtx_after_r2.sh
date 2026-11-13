#!/usr/bin/env bash
# Rig: once the CPU build host built the round-2 5090 binaries, fetch them and run the 5090 correctness
# cells (balanced profile: no --perf; nothing here is timed).
set -u
D=$HOME/.cache/memra-1221b
export PATH="$HOME/.local/bin:$PATH"
waitfor --ssh build-host "grep -qE 'BUILD-DONE|BUILD-FAIL' ~/.cache/memra-1221b/build5090-r2.log" --timeout 300m --every 60s --quiet || exit 1
ssh build-host "grep -q BUILD-DONE ~/.cache/memra-1221b/build5090-r2.log" || { echo "$(date -Is) 5090 R2 build failed" >> "$D/logs/rtx-chain.log"; exit 1; }
mkdir -p "$D/bins5090/R2"
timeout 600 rsync -a build-host:.cache/memra-1221b/bins5090/R2/ "$D/bins5090/R2/" || exit 1
exec bash "$D/tools/rtx_chain.sh" "$D/logs/steps-rtx-r2.txt"
