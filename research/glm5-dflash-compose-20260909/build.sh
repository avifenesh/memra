#!/bin/bash
set -uo pipefail
cd /root/wt-glm5-dflash-compose
source /root/tunebox/env.sh
export CARGO_TARGET_DIR=/root/glm5-dflash-compose-target
R=research/glm5-dflash-compose-20260909
nice -n 19 cargo build --release --bin memra-server -j 16 > "$R/build.log" 2>&1
rc=$?
echo "$rc" > "$R/build.exit"
if [ "$rc" = 0 ]; then
 sha256sum "$CARGO_TARGET_DIR/release/memra-server" > "$R/binary.sha256"
 nice -n 19 cargo test --release -p memra-engine --lib causal_pmin -j 16 -- --test-threads=1 > "$R/causal-tests.log" 2>&1
 echo "$?" > "$R/causal-tests.exit"
fi
exit "$rc"
