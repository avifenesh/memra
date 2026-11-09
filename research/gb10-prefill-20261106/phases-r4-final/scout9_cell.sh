#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: the next lever's scout, the timestamped server-log
# probe (every log line stamped, MEMRA_TICK_TRACE=1) at 32k / 131k / 260k on lever 8b's binary: the
# request's phases after the trunk (the draft fill, the first speculative round, the first token).
set -u
D=$HOME/.cache/memra-gb10-r4
bash "$D/box/ts_probe.sh" "$D/runs/ts-probe-8b" "$D/bins/tft-efbfeb8b10" > "$D/logs/ts-8b.out" 2>&1
echo "SCOUT9-DONE"
