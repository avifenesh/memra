#!/usr/bin/env bash
# Runs ON a GB10 host under the lease after unit_b.sh: the fast-exp probe arms against h2p
# (timing at four depths, both passes) and the ragged byte compare.
set -u
D=$HOME/.cache/memra-gb10-r4
waitfor --grep UNIT-B-DONE "$D/logs/unit_b.txt" --timeout 40m > /dev/null 2>&1
bash "$D/probe/run_probe.sh" ex2-r1 "$D/tree" 4096 "4096,16384,65536,131072" 3 "h2p,ex2,ex2b,h2p,ex2b,ex2" > /dev/null 2>&1
bash "$D/probe/edge.sh" edge-ex2 "h2p,ex2,ex2b" > /dev/null 2>&1
echo UNIT-C-DONE >> "$D/logs/unit_b.txt"
