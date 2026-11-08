#!/usr/bin/env bash
# Runs ON b23d: pause the served campaign at its first cell boundary (lend b23d to #1099's record).
# Waits until cell 2 is announced in campaign.log (cell 1 complete), then stops b23d-fup.sh's
# process group and waits for an idle GPU. Cell 1's data stays; the rest resumes in a second dir.
set -u
D=$HOME/.cache/memra-f8f4v3
C=$D/runs/fup-campaign/served/campaign.log
waitfor --grep 'cell 2 arm=' "$C" --timeout 20m 2>&1 | tail -n 1
pg=$(ps -eo pgid=,args= | awk '/[b]23d-fup\.sh/ {print $1; exit}')
if [ -n "$pg" ]; then echo "stopping process group $pg"; kill -TERM -- "-$pg"; fi
waitfor --until 'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"' --every 2s --timeout 120s 2>&1 | tail -n 1
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
grep -E "cell|^B " "$C" | head -n 8
echo PAUSE-CAMPAIGN-DONE
