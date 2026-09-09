#!/bin/bash
set -uo pipefail
cd /root/wt-glm5-dflash-rootcause
source /root/tunebox/env.sh
export CARGO_TARGET_DIR=/root/glm5-dflash-rootcause-target
nice -n 19 cargo build --release --bin memra-server -j 16 > research/glm5-dflash-rootcause-20260909/build.log 2>&1
rc=$?
echo "$rc" > research/glm5-dflash-rootcause-20260909/build.exit
if [ "$rc" = 0 ]; then sha256sum "$CARGO_TARGET_DIR/release/memra-server" > research/glm5-dflash-rootcause-20260909/binary.sha256; fi
exit "$rc"
