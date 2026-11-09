#!/bin/bash
# The steady campaign on GPU 0 at 8 workers (6 never reach the reclaim: out-w6-stopped/), pinned.
set -u
R=<lane>
cd "$R" && taskset -c 24-31,56-63 bash ./campaign-steady.sh "$R/out"
