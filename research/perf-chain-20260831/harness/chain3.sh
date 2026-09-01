#!/usr/bin/env bash
# GPU program 2: cell 3 (engine staircase + draft-graph flag arms, one fixed env,
# 8 arms interleaved x3), then cell 2's x5 escalation (both amendment rules fired).
set -u
D=/home/ubuntu/perf-chain
B=$D/bin
"$D/harness/run-ab.sh" cell3 "1 2 3" \
  "S04:$B/memra-server-3d52b8531a31:era-nodoors" \
  "S15:$B/memra-server-305876ede4d9:era-nodoors" \
  "S24:$B/memra-server-abc4014151d1:era-nodoors" \
  "S25:$B/memra-server-41b0040e4101:era-nodoors" \
  "S37:$B/memra-server-b3a2d92ff051:era-nodoors" \
  "S44:$B/memra-server-3999a92a6e18:era-nodoors" \
  "FNOFILT:$B/memra-server-3999a92a6e18:fixed-nofiltered" \
  "FNOCHAIN:$B/memra-server-3999a92a6e18:fixed-nochaingraph"
"$D/harness/run-ab.sh" cell2 "4 5" \
  "P:$B/memra-server-3999a92a6e18:current" \
  "PV:$B/memra-server-3999a92a6e18:current-novision"
echo "CHAIN3_DONE $(date -u +%FT%TZ)" >> "$D/receipts/progress-cell3.txt"
