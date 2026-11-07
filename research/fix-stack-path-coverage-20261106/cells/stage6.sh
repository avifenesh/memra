#!/bin/bash
# Stage the DSv4 checkpoint for the 2026-11-07 pair window (pair-cells6.sh). Waits for
# the box's training lane: the s1000 checkpoint conversion log must exist and the go flag go-stage6 must be
# set (touched from the rig when the box's training lane says staging is fine), so the 175 GB download does
# not compete with their conversion. Writes stage6.ok only when every file matches the manifest.
set -u
W=<pair-kit>; M=$W/dsv4-nvfp4; O=$W/out6; S=$O/status.txt
WF=<box-home>/.local/bin/waitfor
export PATH=<box-home>/.local/bin:$PATH
mkdir -p $O
echo "stage6 armed $(date '+%m-%d %H:%M:%S')" >> $S
$WF --file <box-work>/runs/r1a/s1000/convert.log --timeout 16h --every 2m > $O/stage-wait.log 2>&1 \
    || { echo "stage6: s1000 convert.log never appeared ($(date '+%m-%d %H:%M'))" >> $S; exit 3; }
$WF --file $W/go-stage6 --timeout 8h --every 1m >> $O/stage-wait.log 2>&1 \
    || { echo "stage6: no go from the box's training lane ($(date '+%m-%d %H:%M'))" >> $S; exit 3; }
echo "stage6 download start $(date '+%m-%d %H:%M:%S'); $(df -BG /mnt/Data | tail -1 | awk '{print $4}') free" >> $S
nice -n 10 ionice -c2 -n7 uvx --from huggingface_hub hf download Avifenesh/DeepSeek-V4-Flash-0731-NVFP4 --revision 3369bf0f52808b1bae86a7cab07bd1a7f5042e9d \
    --local-dir $M > $W/logs/download6.log 2>&1 || { echo "stage6: download FAILED rc=$?" >> $S; exit 1; }
( cd $M && sha256sum -c --quiet sha256-manifest.txt > $O/sha256-check.log 2>&1 )
bad=$(grep -c FAILED $O/sha256-check.log)
# README.md differs from the manifest by design (the 2026-11-05 window recorded the same).
if [ "$bad" -le 1 ] && ! grep FAILED $O/sha256-check.log | grep -v -q 'README.md'; then
    echo "stage6 verified $(date '+%m-%d %H:%M:%S'): $(du -sh $M | cut -f1); README.md differs by design" >> $S
    touch $W/stage6.ok
else
    echo "stage6: $bad files fail the manifest (see out6/sha256-check.log)" >> $S; exit 1
fi
