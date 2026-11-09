#!/bin/bash
# Partial-reclaim lane, final run: build A and B from one fresh clone and target dir, then the campaign on GPU 0.
set -u
R=<lane>
bash "$R/build-arm.sh" main d35337f51add9b16033e42c573c3fc2b4a8634ca || exit 2
bash "$R/build-arm.sh" fix/partial-reclaim-discrete ab21e254a6e6d97cf681c13296596d571a94f007 || exit 3
cd "$R" && taskset -c 24-31,56-63 bash ./campaign.sh 0 "$R/out"
