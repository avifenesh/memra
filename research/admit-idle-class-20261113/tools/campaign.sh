#!/usr/bin/env bash
# memra#1304, runs ON GB10 cb33 inside the caller's sparklock lease: every cell of the campaign in
# order, one fresh server each (cell.sh). M = main 01960ee1b1, H = the fix branch.
#   01-M, 02-H  the issue's recipe: MemAvailable to about 1.8 GB, one g1k request
#   03-H, 04-M  MemAvailable to about 1.5 GB: short even with nothing booked
#   05-M, 06-H  MemAvailable to about 2.4 GB: short only with the cache's growth booked
#   07-H, 08-M  ample memory (no holder): admission unchanged
#   09-M, 10-H  MemAvailable to about 20 GB, a g28k request then a g1k one while it primes
#   11-H        ample memory, MEMRA_ADMIT_RESERVE_MB=200000 (the teeth door): a charge above the
#               post-load capacity, so the 400 path
# usage: campaign.sh CAMPAIGN
set -u
D=$HOME/.cache/memra-1304; C=$1
B=$D/box/cell.sh
bash "$B" "$C" 01-M main 1800
bash "$B" "$C" 02-H head 1800
bash "$B" "$C" 03-H head 1500
bash "$B" "$C" 04-M main 1500
bash "$B" "$C" 05-M main 2400
bash "$B" "$C" 06-H head 2400
bash "$B" "$C" 07-H head 0
bash "$B" "$C" 08-M main 0
MODE=pair bash "$B" "$C" 09-M main 20000
MODE=pair bash "$B" "$C" 10-H head 20000
EXTRA="MEMRA_ADMIT_RESERVE_MB=200000" bash "$B" "$C" 11-H head 0
echo "CAMPAIGN-DONE $(date -Is)" >> "$D/runs/$C/campaign.log"
