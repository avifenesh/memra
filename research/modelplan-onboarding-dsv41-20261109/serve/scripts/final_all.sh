#!/bin/bash
# The lane's final one-card sequence on the committed binaries, one lease per step, stopping at a step boundary when
# $ROOT/serve/STOP or the box's QUIET marker appears: the residency gate (dsv41-run only), CheckpointParity on the
# residency, the serve gate, the chat gates, the lib suites, the shared benchmark client; then the step rates and the
# receipts gathered. `final_all.sh skip-res` keeps an earlier residency gate of the same dsv41-run binary.
. "${LANE_ENV:?LANE_ENV names the machine file that sets ROOT GPU_UUID BOX_LOCK QUIET_FILE CORES GPU_RUN BUILD}"
set -u
S=$ROOT/serve
cd $S
if [ "${1:-}" != skip-res ]; then
  ./scripts/res_gate.sh > $S/res_gate.out 2>&1; echo "res_gate rc $?"
  [ -e $S/STOP ] && { echo stopped; exit 4; }
fi
./scripts/serve_gate.sh all > $S/serve_gate_all.out 2>&1; echo "serve_gate rc $?"
[ -e $S/STOP ] && { echo stopped; exit 4; }
./scripts/test_serve.sh run > $S/test_run.out 2>&1; echo "test_serve rc $?"
[ -e $S/STOP ] && { echo stopped; exit 4; }
./scripts/serve_gate.sh bench > $S/serve_gate_bench.out 2>&1; echo "bench rc $?"
python3 $S/scripts/rates.py $S/logs/serve-chat.log $S/rates.json
rm -rf $S/bank
./scripts/bank.sh $S/bank
echo FINAL-ALL-DONE
