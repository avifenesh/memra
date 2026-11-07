#!/usr/bin/env bash
set -u
D=$HOME/.cache/memra-attn-gb10
bash "$D/probe/run_probe.sh" r2 4096 "4096,16384,65536,131072" 3 "h2,pp,pp2,ppad,ppadm,ppm,qh12,sbad" > /dev/null 2>&1
bash "$D/probe/ncu_probe.sh" r2-ppad ppad 4096 65536 > "$D/logs/ncu-r2-ppad.log" 2>&1
echo ROUND2-DONE > "$D/logs/round2.done"
