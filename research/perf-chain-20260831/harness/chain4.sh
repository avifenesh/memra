#!/usr/bin/env bash
# GPU program 3 (anchor cell). The bench box rebooted between cell 1 and cell 3, so cell 1's
# OD arm (the range's LEFT endpoint under the fixed env) and cell 3's staircase sit in
# different kernel sessions. This cell re-measures the left endpoint IN THE SAME SESSION as
# the two staircase ends, interleaved, so the "what did the 82 engine commits do" claim never
# crosses a reboot:
#   ODX = era commit c9a617ca994b, fixed env   (left endpoint, re-anchored)
#   S04 = 3d52b8531a31, fixed env              (first staircase step)
#   S44 = 3999a92a6e18, fixed env              (right endpoint)
set -u
D=/home/ubuntu/perf-chain
B=$D/bin
for i in $(seq 1 960); do
  grep -q CHAIN3_DONE "$D/receipts/progress-cell3.txt" 2>/dev/null && break
  sleep 15
done
grep -q CHAIN3_DONE "$D/receipts/progress-cell3.txt" 2>/dev/null || { echo "WATCH_TIMEOUT"; exit 1; }
"$D/harness/run-ab.sh" anchor "1 2 3" \
  "ODX:$B/memra-server-c9a617ca994b:era-nodoors" \
  "S04A:$B/memra-server-3d52b8531a31:era-nodoors" \
  "S44A:$B/memra-server-3999a92a6e18:era-nodoors"
echo "CHAIN4_DONE $(date -u +%FT%TZ)" >> "$D/receipts/progress-anchor.txt"
