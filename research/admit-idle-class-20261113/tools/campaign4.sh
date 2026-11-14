#!/usr/bin/env bash
# memra#1304 review fold, runs ON the GB10 unit inside the caller's sparklock lease, after
# campaign3.sh: the DFlash short-memory cell with a holder that does not give memory back.
#   31-G  the DFlash2 drafter, one g131k request, MemAvailable taken to 28-G's charge plus
#         MARGIN_MB (default 1500) by a static holder (mem_holder.py static)
# campaign3's 29-G used the regulating holder, which returned memory as the server grew: the unit
# stayed at the target through the prime, 12.1 GB was free at the publication and the copy fit.
# Memory another process holds does not shrink for the server; here the request's own KV comes
# out of what is left, and the 4.1 GB copy must reserve room that is not there. The margin keeps
# MemAvailable above the 1 GiB guard at the prime's peak (the pool grew 9.7 GB in 28-G).
# usage: campaign4.sh CAMPAIGN
set -u
D=$HOME/.cache/memra-1304; C=$1
B=$D/box/cell.sh
R=$D/runs/$C
tgt=$(python3 "$D/box/charge.py" "$R/28-G/server.log" "${MARGIN_MB:-1500}" 2>> "$R/campaign.log")
if [ -n "$tgt" ]; then
  echo "31-G target ${tgt} MiB, static holder ($(python3 "$D/box/charge.py" "$R/28-G/server.log" "${MARGIN_MB:-1500}" --explain))" >> "$R/campaign.log"
  ROUTE=dflash REQ=131072 HOLD_S=900 SETTLE_MB=48 HOLD_MODE=static bash "$B" "$C" 31-G fold "$tgt"
else
  echo "31-G SKIPPED: no charge read from 28-G" >> "$R/campaign.log"
fi
echo "CAMPAIGN4-DONE $(date -Is)" >> "$R/campaign.log"
