#!/usr/bin/env bash
set -u
D=$HOME/.cache/memra-attn-gb10
bash "$D/probe/run_probe.sh" r3 4096 "4096,16384,65536,131072" 3 "h2,ad,sbad,ppad,ppadm,ppadz,h2" > /dev/null 2>&1
bash "$D/probe/ncu_probe.sh" r3-ppad ppad 4096 65536 > "$D/logs/ncu-r3-ppad.log" 2>&1
echo ROUND3-DONE > "$D/logs/round3.done"
