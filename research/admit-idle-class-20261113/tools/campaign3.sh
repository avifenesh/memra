#!/usr/bin/env bash
# memra#1304 review fold, runs ON the GB10 unit inside the caller's sparklock lease: the cells of the
# fold build (G = arm `fold`), one fresh server each (cell.sh).
#   18-G .. 23-G  the head's idle-admit and refusal cells again (campaign2.sh's 12-F .. 17-F), on
#                 the MTP route: 1.8 GB and 1.5 GB (503), 2.4 GB (admitted), ample, the 20 GB pair,
#                 the teeth door (400)
#   24-G          the DFlash2 drafter (the DSpark route, whose publisher copies), ample memory, one
#                 g131k request: the reference text, the publication and the request's charge
#   25-G          the same server and request with MemAvailable held just above that charge: the
#                 arrival is admitted with nothing booked, and its publication must reserve room
#                 it does not have
# 25-G's target is computed from 24-G's server log: the request cost, the per-session draft-state
# charge and the transient reserve the boot calibration states, plus MARGIN_MB (default 600), in
# MiB as mem_holder.py takes it.
# usage: campaign3.sh CAMPAIGN
set -u
D=$HOME/.cache/memra-1304; C=$1
B=$D/box/cell.sh
R=$D/runs/$C
bash "$B" "$C" 18-G fold 1800
bash "$B" "$C" 19-G fold 1500
bash "$B" "$C" 20-G fold 2400
bash "$B" "$C" 21-G fold 0
MODE=pair bash "$B" "$C" 22-G fold 20000
EXTRA="MEMRA_ADMIT_RESERVE_MB=200000" bash "$B" "$C" 23-G fold 0
ROUTE=dflash REQ=131072 bash "$B" "$C" 24-G fold 0
tgt=$(python3 "$D/box/charge.py" "$R/24-G/server.log" "${MARGIN_MB:-600}" 2>> "$R/campaign.log")
if [ -n "$tgt" ]; then
  echo "25-G target ${tgt} MiB ($(python3 "$D/box/charge.py" "$R/24-G/server.log" "${MARGIN_MB:-600}" --explain))" >> "$R/campaign.log"
  ROUTE=dflash REQ=131072 HOLD_S=900 bash "$B" "$C" 25-G fold "$tgt"
  # A target under the charge the server computes gets the 503, which states that charge: one
  # more cell from it.
  need=$(grep -o 'idle memory short: .*required [0-9]*MB' "$R/25-G/server.log" | grep -o 'required [0-9]*MB' | grep -o '[0-9]*' | tail -n 1)
  if [ -n "$need" ]; then
    tgt2=$(( (need + ${MARGIN_MB:-600}) * 1000000 / 1048576 ))
    echo "25-G was refused (required ${need} MB); 26-G target ${tgt2} MiB" >> "$R/campaign.log"
    ROUTE=dflash REQ=131072 HOLD_S=900 bash "$B" "$C" 26-G fold "$tgt2"
  fi
else
  echo "25-G SKIPPED: no charge read from 24-G" >> "$R/campaign.log"
fi
echo "CAMPAIGN3-DONE $(date -Is)" >> "$R/campaign.log"
# Added while the campaign ran: 18-G sent its request 26 s into the holder's descent, at
# MemAvailable 1,994 MiB (cell.sh waited only until it was within 300 MiB of the target), which is
# above the charge, so it was admitted. The 1.8 GB refusal cell again, the request held until the
# holder is within 48 MiB of its target.
SETTLE_MB=48 bash "$B" "$C" 27-G fold 1800
echo "CAMPAIGN3-EXTRA-DONE $(date -Is)" >> "$R/campaign.log"
# Added while the campaign ran: 24-G did not boot. serve_short.sh's DFlash arm still set
# MEMRA_DSPARK_PARTIAL_RESTORE, retired on 2026-11-07, and the boot audit refused it (no request
# was sent; 25-G was skipped). The two DFlash cells again with the arm at the served defaults.
ROUTE=dflash REQ=131072 bash "$B" "$C" 28-G fold 0
tgt=$(python3 "$D/box/charge.py" "$R/28-G/server.log" "${MARGIN_MB:-600}" 2>> "$R/campaign.log")
if [ -n "$tgt" ]; then
  echo "29-G target ${tgt} MiB ($(python3 "$D/box/charge.py" "$R/28-G/server.log" "${MARGIN_MB:-600}" --explain))" >> "$R/campaign.log"
  ROUTE=dflash REQ=131072 HOLD_S=900 bash "$B" "$C" 29-G fold "$tgt"
  need=$(grep -o 'idle memory short: .*required [0-9]*MB' "$R/29-G/server.log" | grep -o 'required [0-9]*MB' | grep -o '[0-9]*' | tail -n 1)
  if [ -n "$need" ]; then
    tgt2=$(( (need + ${MARGIN_MB:-600}) * 1000000 / 1048576 ))
    echo "29-G was refused (required ${need} MB); 30-G target ${tgt2} MiB" >> "$R/campaign.log"
    ROUTE=dflash REQ=131072 HOLD_S=900 SETTLE_MB=48 bash "$B" "$C" 30-G fold "$tgt2"
  fi
else
  echo "29-G SKIPPED: no charge read from 28-G" >> "$R/campaign.log"
fi
echo "CAMPAIGN3-DFLASH-DONE $(date -Is)" >> "$R/campaign.log"
