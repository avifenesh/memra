#!/bin/bash
# The owner's "one battery cell" for the DSpark strict-prefix restore pair (S17:F6), on the RTX 5090,
# with the binary that still has the pair (fe262e905b). Restored-vs-cold byte identity on a 3-turn chat:
#   WARM: one boot, MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_PARTIAL_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1,
#         turns 1, 2, 3 in order (turns 2 and 3 strictly extend the previous prompt).
#   COLD: one fresh boot per turn-2 and turn-3 request, the pair and prefix restore unset (monolithic prime).
# PASS = every warm answer byte-equal to its cold twin, and the warm boot logs a strict-prefix DSPARK restore
# (restored tokens < prompt tokens) on turns 2 and 3.
EV=${EV:?evidence dir}; mkdir -p "$EV"
BIN=${BIN:?memra-server binary}
MODEL=${MODEL:?Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
DRAFT=${DRAFT:?DFlash2 drafter export dir}
PORT=${PORT:-18171}
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
sha256sum $BIN | cut -c1-16 > $EV/binary.sha
COMMON=(MEMRA_COMPAT=openai "MEMRA_MODELS=q=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=2
        MEMRA_DSPARK_SPEC=1 "MEMRA_DSPARK_DRAFT=$DRAFT")
# WARM_ENV overrides the warm boot's switches; unset it to repeat the 2026-11-04 run.
read -r -a PAIR <<< "${WARM_ENV:-MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_PARTIAL_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1}"

boot() {  # boot LOG ENV...
    local log=$1; shift
    env "${COMMON[@]}" "$@" $BIN > $log 2>&1 &
    SPID=$!
    waitfor --http http://127.0.0.1:$PORT/v1/models --timeout 10m > /dev/null 2>&1 || { echo "boot failed: $(tail -n 3 $log)"; return 1; }
}
stop() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }
ask() { python3 "$HERE/dspark-pair-client.py" $PORT "$@"; }

echo "== warm boot $(date +%H:%M:%S)" | tee $EV/run.txt
boot $EV/warm-server.log "${PAIR[@]}" || exit 1
ask turn 1 $EV/warm-t1.json && ask turn 2 $EV/warm-t2.json $EV/warm-t1.json && ask turn 3 $EV/warm-t3.json $EV/warm-t1.json $EV/warm-t2.json
stop
for t in 2 3; do
    echo "== cold boot for turn $t $(date +%H:%M:%S)" | tee -a $EV/run.txt
    boot $EV/cold-t$t-server.log || exit 1
    if [ $t = 2 ]; then ask turn 2 $EV/cold-t2.json $EV/warm-t1.json; else ask turn 3 $EV/cold-t3.json $EV/warm-t1.json $EV/warm-t2.json; fi
    stop
done
grep -a -E '\[prefix-cache\] DSPARK restore|insert probation|prompt=[0-9]+ cached=' $EV/warm-server.log | cut -c1-220 > $EV/warm-restore-lines.txt
python3 "$HERE/dspark-pair-client.py" compare $EV | tee -a $EV/run.txt
