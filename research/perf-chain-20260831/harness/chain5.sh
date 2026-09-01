#!/usr/bin/env bash
# GPU program 4: x5 escalation for the arms whose verdicts are load-bearing.
# Amendment rule (1) fired on every arm in this lane (all within-arm spreads exceed 0.5%),
# and rule (2) fired on the flag arms (their deltas sit inside 2x the pooled spread). One
# rotation carries both groups, so the anchor endpoints and the two draft-graph flag arms
# stay interleaved with the pin they are compared against:
#   ODX/S04A/S44A  -> the anchor cell's endpoints (positive +6.4% wall / flat-decode claim)
#   FNOFILT/FNOCHAIN -> the draft-graph door arms, compared against S44A (same binary+env)
# Boots 4 and 5; arm names match the earlier cells so the rows pool by name.
set -u
D=/home/ubuntu/perf-chain
B=$D/bin
"$D/harness/run-ab.sh" esc "4 5" \
  "ODX:$B/memra-server-c9a617ca994b:era-nodoors" \
  "S04A:$B/memra-server-3d52b8531a31:era-nodoors" \
  "S44A:$B/memra-server-3999a92a6e18:era-nodoors" \
  "FNOFILT:$B/memra-server-3999a92a6e18:fixed-nofiltered" \
  "FNOCHAIN:$B/memra-server-3999a92a6e18:fixed-nochaingraph"
echo "CHAIN5_DONE $(date -u +%FT%TZ)" >> "$D/receipts/progress-esc.txt"
