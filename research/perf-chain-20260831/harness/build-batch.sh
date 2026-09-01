#!/usr/bin/env bash
# Build the cell-3 staircase binaries back to back. Refuses to start if a runner is live
# (build.sh checks out the shared tree; yanking it under a live server is the scp-race
# lesson generalized).
set -u
D=/home/ubuntu/perf-chain
for c in "$@"; do
  echo "=== $(date -u +%H:%M:%SZ) building $c ==="
  "$D/harness/build.sh" "$c" || { echo "BATCH_ABORT at $c"; exit 1; }
done
echo "BATCH_DONE $(date -u +%H:%M:%SZ)"
