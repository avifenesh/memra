#!/usr/bin/env bash
# round 4: the shipped fa_prefill_qw_h2p next to h2 and its probe copy, two skew variants and
# QK(i+1)-before-softmax(i), then the ragged shapes on the shipped symbol.
set -u
D=$HOME/.cache/memra-attn-gb10
bash "$D/probe/run_probe.sh" r4 4096 "4096,16384,65536,131072" 3 "h2,h2p,ppad,sk1,sk2,pqk,h2,h2p" > /dev/null 2>&1
bash "$D/probe/edge.sh" edge-h2p "h2,h2p" > /dev/null 2>&1
echo ROUND4-DONE > "$D/logs/round4.done"
