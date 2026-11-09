#!/bin/bash
# After the GPU 0 acceptance campaign: the waves cell B A A B on GPU 0 (a second-card replicate).
set -u
R=<lane>
<home>/.local/bin/waitfor --until "grep -q 'reclaim: done' $R/out/campaign.log" --timeout 60m --every 20s --quiet || exit 4
cd "$R" && taskset -c 24-31,56-63 bash ./campaign-ab.sh 0 "$R/out-gpu0w" waves "B A A B"
