#!/bin/bash
# After the GPU 1 mixed run: the waves cell A B B A on GPU 1.
set -u
R=<lane>
<home>/.local/bin/waitfor --until "grep -q 'reclaim-gpu1: done' $R/out-gpu1/campaign.log" --timeout 30m --every 15s --quiet || exit 4
cd "$R" && taskset -c 24-31,56-63 bash ./campaign-ab.sh 1 "$R/out-gpu1" waves "A B B A"
