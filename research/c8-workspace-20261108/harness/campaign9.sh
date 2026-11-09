#!/usr/bin/env bash
# After the prime-pass A/B (campaign8): agentic c8 with A = S and B = C1g, then c4 and c1, one
# lease each (2026-11-09 from 05:50).
#   S=.. C1G=.. campaign9.sh
set -u
H=<harness>
y() { [ -e <lane>/YIELD ] && { echo "yield"; exit 0; }; }
export A=$S B=$C1G
y; bash "$H/group2.sh" "c8g|0|A B B A A B|agentic:8:1.0:768|"
y; bash "$H/group2.sh" "c4|0|A B B A A B|agentic:4:1.0:768|" "c1|0|A B|agentic:1:1.0:768|"
echo CAMPAIGN9_DONE
