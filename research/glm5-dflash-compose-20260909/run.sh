#!/bin/bash
set -euo pipefail
cd /root/wt-glm5-dflash-compose/research/glm5-dflash-compose-20260909
while [ ! -f build.exit ] || [ ! -f causal-tests.exit ]; do
 if [ -f build.exit ] && [ "$(cat build.exit)" != 0 ]; then exit 1; fi
 sleep 5
done
[ "$(cat build.exit)" = 0 ]
[ "$(cat causal-tests.exit)" = 0 ]
exec flock /tmp/memra-gpu.lock nice -n 19 python3 cell.py
