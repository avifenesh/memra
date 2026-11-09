#!/usr/bin/env bash
# The prime-pass tail on the keyed card (2026-11-09 from 05:00): the agentic c8 prime pass alone,
# C1e (A) against C1g (B, the deadline estimate that keeps the tick takes), A B B A A B in one lease; then, with
# A = S and B = C1g, agentic c8, c4 and c1, one lease each.
#   S=.. C1E=.. C1G=.. campaign8.sh
set -u
H=<harness>
y() { [ -e <lane>/YIELD ] && { echo "yield"; exit 0; }; }
y; A=$C1E B=$C1G bash "$H/group2.sh" "prime|0|A B B A A B|agentic:8:1.0:768:prime|"
echo PRIME_DONE
[ -e <lane>/STOP_AFTER_PRIME ] && { echo "stop after prime"; exit 0; }
export A=$S B=$C1G
y; bash "$H/group2.sh" "c8g|0|A B B A A B|agentic:8:1.0:768|"
y; bash "$H/group2.sh" "c4|0|A B B A A B|agentic:4:1.0:768|" "c1|0|A B|agentic:1:1.0:768|"
echo CAMPAIGN8_DONE
