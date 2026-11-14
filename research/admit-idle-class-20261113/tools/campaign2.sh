#!/usr/bin/env bash
# memra#1304, runs ON GB10 cb33 inside the caller's sparklock lease: the head cells again on the
# final head binary (bins/final: the head with one `let` moved above `let drained`, so the drafter
# wiring pin reads its contiguous source again; the same program). Same conditions as campaign.sh.
# usage: campaign2.sh CAMPAIGN
set -u
D=$HOME/.cache/memra-1304; C=$1
B=$D/box/cell.sh
bash "$B" "$C" 12-F final 1800
bash "$B" "$C" 13-F final 1500
bash "$B" "$C" 14-F final 2400
bash "$B" "$C" 15-F final 0
MODE=pair bash "$B" "$C" 16-F final 20000
EXTRA="MEMRA_ADMIT_RESERVE_MB=200000" bash "$B" "$C" 17-F final 0
echo "CAMPAIGN2-DONE $(date -Is)" >> "$D/runs/$C/campaign.log"
