#!/usr/bin/env bash
# Cell 3: the engine staircase + the draft-graph flag arms, ALL under the one fixed env
# family (era-nodoors), interleaved together x3 so every pair in the rotation is a
# fresh-boot interleaved A/B. Left endpoint of the staircase is cell 1's OD arm (same
# binary, same env). S24 vs S25 is the direct boundary A/B at the draft-graph merge.
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
echo "CHAIN2_DONE $(date -u +%FT%TZ)" >> "$D/receipts/progress-cell3.txt"
