#!/bin/bash
# Hard stop for pair-cells6.sh and stage6.sh: at 11:00 on 2026-11-07 kill anything run from bin6/ or
# tools6/ and the two scripts, then delete the staged model and drafter (bin6/ and tools6/ stay:
# 300 MB, reused by the quiet slot).
W=<pair-kit>; O=$W/out6; mkdir -p $O
WF=<box-home>/.local/bin/waitfor
STOP=$(date -d '2026-11-07 11:00' +%s)
$WF --until "test \$(date +%s) -ge $STOP" --timeout 20h --every 60s > $O/guard-wait.log 2>&1
if [ "$(date +%s)" -lt "$STOP" ]; then
  echo "guard6 wait ended early at $(date '+%m-%d %H:%M:%S') ($(tail -n 1 $O/guard-wait.log)); not cleaning" >> $O/status.txt; exit 1
fi
pkill -TERM -f 'bash <pair-kit>/pair-cells6.sh' 2>/dev/null
pkill -TERM -f 'bash <pair-kit>/stage6.sh' 2>/dev/null
pkill -TERM -f '<pair-kit>/bin6/' 2>/dev/null
pkill -TERM -f '<pair-kit>/tools6/' 2>/dev/null
pkill -TERM -f 'hf download Avifenesh/DeepSeek-V4-Flash-0731-NVFP4' 2>/dev/null
$WF --no-proc '<pair-kit>/bin6/' --timeout 60s >> $O/guard-wait.log 2>&1
pkill -9 -f '<pair-kit>/bin6/' 2>/dev/null
rm -rf <pair-kit>/dsv4-nvfp4 <pair-kit>/dflash2-q38 <pair-kit>/stage6.ok
echo "guard6 $(date '+%m-%d %H:%M:%S') stopped and cleaned; $(df -BG /mnt/Data | tail -1 | awk '{print $4}') free; gpu0,1 MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | tr '\n' ' ')" >> $O/status.txt
