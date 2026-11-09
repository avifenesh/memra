#!/bin/bash
# After build-ab.sh: the steady campaign on GPU 0, pinned.
set -u
R=<lane>
<home>/.local/bin/waitfor --until "grep -q 'builds done\|FAILED' $R/status.txt" --timeout 60m --every 15s --quiet || exit 4
grep -q FAILED "$R/status.txt" && exit 5
cd "$R" && taskset -c 24-31,56-63 bash ./campaign-steady.sh "$R/out"
