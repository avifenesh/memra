#!/bin/bash
# Follow-up to gpu-1106b on tip 99c65ab0b2: the cells whose first run never reached their path.
#   floor-ab      Q38 + DFlash2 at the default session cap refused a 329-token request on an idle 5090
#                 (transient floor 4808MB); old binary vs tip vs tip at 4 sessions.
#   dflash-demote rerun at MEMRA_MAX_SESSIONS=4 (the smoke's cap).
#   mtp-cohort    c=8 is far concurrency, so the cost model served it plain; c=2/c=4 waves and K pinned.
#   cold-drafter  the gate's default prompts (about 26k tokens) exceed the 27B's 18.9k on 24 GB; sized
#                 30/30, with the gate's exit status and path counters fixed (cells-1106 copy).
set -u
W=<work>/engine-dsv4-glm
cd $W || exit 2
L=<work>/logs-gpu-1106c; mkdir -p $L; S=$L/status
C=<work>/cells-1106
L1=<work>/logs-gpu-1106
BIN=$L1/memra-server-99c65ab0b20e
BIN_OLD=<work4>/logs-top/memra-server-196596e5d7e3
G() { local ttl=$1 why=$2; shift 2; gpulock run --ttl "$ttl" --for "memra tip 1106 path cells, follow-up: $why" --wait 1h -- "$@"; }
mkdir -p $L/serve; cp <work>/logs-gpu-1106b/serve/mtp-solo.json $L/serve/
echo "start $(date +%H:%M)" > $S
for sc in floor-ab dflash-demote mtp-cohort; do
  G 40m "serve cells $sc" python3 $C/serve_cells.py $sc $BIN $L/serve $BIN_OLD > $L/serve-$sc.log 2>&1
  echo "serve-$sc rc=$? $(grep -a 'CELL' $L/serve-$sc.log | cut -c1-260 | tr '\n' ' ')" >> $S
done
G 1h "DSpark cold-drafter gate, 5090 sizes" env GATE_P_PARAS=30 GATE_S_PARAS=30 GATE=$W/tools/dspark-cold-drafter-gate.py \
  bash $C/cold-drafter-gate.sh $BIN $L/cold-drafter > $L/cold-drafter.log 2>&1
echo "cold-drafter rc=$? $(grep -a -E 'identical=|GATE|=[0-9]+$|never logged|FAILED' $L/cold-drafter.log | tr '\n' ' ' | cut -c1-400)" >> $S
echo "done $(date +%H:%M)" >> $S
