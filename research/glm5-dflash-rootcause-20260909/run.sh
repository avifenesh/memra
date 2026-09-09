#!/bin/bash
set -euo pipefail
cd /root/wt-glm5-dflash-rootcause/research/glm5-dflash-rootcause-20260909
while [ ! -f build.exit ]; do sleep 5; done
[ "$(cat build.exit)" = 0 ]
flock /tmp/memra-gpu.lock python3 cell.py > cell.log 2>&1
