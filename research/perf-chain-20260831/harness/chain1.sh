#!/usr/bin/env bash
# Chained GPU program: cell1 x5 escalation, then cell2 (vision residency) x3.
# Kept in one driver so the cards never idle between cells.
set -u
D=/home/ubuntu/perf-chain
ERA=$D/bin/memra-server-c9a617ca994b
PIN=$D/bin/memra-server-3999a92a6e18
"$D/harness/run-ab.sh" cell1 "4 5" "O:$ERA:era" "OD:$ERA:era-nodoors"
"$D/harness/run-ab.sh" cell2 "1 2 3" "P:$PIN:current" "PV:$PIN:current-novision"
echo "CHAIN1_DONE $(date -u +%FT%TZ)" >> "$D/receipts/progress-cell2.txt"
