#!/usr/bin/env bash
# memra#1304 second review fold, runs ON the GB10 unit inside the caller's sparklock lease: the
# cells of the second fold's build (J = arm `fold`), one fresh server each (cell.sh), then the
# host-tier identity gate.
#   32-J .. 37-J  the branch's idle-admit and refusal cells again, on the MTP route: 1.8 GB and
#                 1.5 GB (503), 2.4 GB (admitted), ample, the 20 GB pair, the teeth door (400)
#   38-J          the DFlash2 drafter, ample memory, one g131k request: the copy is published
#   39-J          the same with a fixed amount of other memory held (mem_holder.py static) at
#                 38-J's charge plus MARGIN_MB (default 1500): the copy must be refused before
#                 it allocates, and the text must be 38-J's
#   host-gate-*   tools/kv-host-spill-identity-gate.sh on the same binary, its default arm and
#                 its drafter arm: an entry is demoted to the host tier and promoted back
#                 through the unified-memory reservation (`host_promote_reserve`), and the round
#                 trip must not change a byte
# usage: campaign5.sh CAMPAIGN
set -u
D=$HOME/.cache/memra-1304; C=$1
B=$D/box/cell.sh
R=$D/runs/$C
SETTLE_MB=48 bash "$B" "$C" 32-J fold 1800
SETTLE_MB=48 bash "$B" "$C" 33-J fold 1500
SETTLE_MB=48 bash "$B" "$C" 34-J fold 2400
bash "$B" "$C" 35-J fold 0
MODE=pair bash "$B" "$C" 36-J fold 20000
EXTRA="MEMRA_ADMIT_RESERVE_MB=200000" bash "$B" "$C" 37-J fold 0
ROUTE=dflash REQ=131072 bash "$B" "$C" 38-J fold 0
tgt=$(python3 "$D/box/charge.py" "$R/38-J/server.log" "${MARGIN_MB:-1500}" 2>> "$R/campaign.log")
if [ -n "$tgt" ]; then
  echo "39-J target ${tgt} MiB, static holder ($(python3 "$D/box/charge.py" "$R/38-J/server.log" "${MARGIN_MB:-1500}" --explain))" >> "$R/campaign.log"
  ROUTE=dflash REQ=131072 HOLD_S=900 SETTLE_MB=48 HOLD_MODE=static bash "$B" "$C" 39-J fold "$tgt"
else
  echo "39-J SKIPPED: no charge read from 38-J" >> "$R/campaign.log"
fi
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
# The gate's two arms: the model's own route (the MTP head), and its drafter arm (the DFlash2
# drafter, whose entries are copies and carry the drafter tail).
for arm in mtp dflash; do
  mkdir -p "$R/host-gate-$arm"
  t0=$(date '+%F %T')
  if [ "$arm" = dflash ]; then
    export MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_PREFIX_RESTORE=1 \
      MEMRA_DSPARK_DRAFT="$HOME/models/memra-q38/Qwen3.8-27B-DFlash2-memra" \
      MEMRA_FRSPEC_TRIM="$G8/q38-ranks-sxc32768.gguf.txt"
  fi
  ( cd "$D/tree-fold" && bash tools/kv-host-spill-identity-gate.sh "$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" "$D/bins/fold/memra-server" "$R/host-gate-$arm/ev" ) > "$R/host-gate-$arm/gate.log" 2>&1
  echo "HOST-GATE $arm rc=$? $(tail -n 1 "$R/host-gate-$arm/gate.log" | cut -c1-200)" >> "$R/campaign.log"
  journalctl -k --since "$t0" --no-pager > "$R/host-gate-$arm/kernel.log" 2>&1
done
echo "CAMPAIGN5-DONE $(date -Is)" >> "$R/campaign.log"
