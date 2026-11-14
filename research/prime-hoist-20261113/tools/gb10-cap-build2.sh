#!/usr/bin/env bash
# Runs ON a GB10 unit (memra#1258), CPU only, inside one sparklock lease: gb10-cap-build.sh for
# each arm named, in order; stops at the first failure. usage: gb10-cap-build2.sh ARM...
set -u
R=$HOME/.cache/memra-1258
for arm in "$@"; do
  if [ -x "$R/bins/memra-server-$arm-121a" ] && grep -q "CAP-BUILD-DONE $arm rc=0" "$R/logs/cap-build-$arm.log" 2>/dev/null; then
    echo "already built: $arm"; continue
  fi
  bash "$R/gb10-cap-build.sh" "$arm" || exit 1
done
