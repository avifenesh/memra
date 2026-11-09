#!/usr/bin/env bash
# The rest after #1117's landing record (2026-11-09): c1 A B, then the DFlash-route grow replay.
#   M=.. L=.. campaign12.sh
set -u
H=<harness>
y() { [ -e <lane>/YIELD ] && { echo "yield"; exit 0; }; }
export A=$M B=$L
y; bash "$H/group2.sh" "c1|0|A B|agentic:1:1.0:768|"
y; BASE=$M FIX=$L bash "$H/grow_check.sh"
echo CAMPAIGN12_DONE
