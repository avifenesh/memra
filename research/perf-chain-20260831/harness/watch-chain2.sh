#!/usr/bin/env bash
# Wait for chain1 (cell1 escalation + cell2) to finish, then start cell 3 immediately.
set -u
D=/home/ubuntu/perf-chain
for i in $(seq 1 720); do
  grep -q CHAIN1_DONE "$D/receipts/progress-cell2.txt" 2>/dev/null && break
  sleep 15
done
grep -q CHAIN1_DONE "$D/receipts/progress-cell2.txt" 2>/dev/null || { echo "WATCH_TIMEOUT"; exit 1; }
exec "$D/harness/chain2.sh"
