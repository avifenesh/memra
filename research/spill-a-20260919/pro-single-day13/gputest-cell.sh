#!/usr/bin/env bash
# The arm-honoured unit cell on the card: pinned_kind_arm_is_honoured_by_the_driver (ignored
# without a device), the release test binary built by build.sh, under the inherited collector lock.
# usage: gputest-cell.sh <lockfd>
set -uo pipefail
fd=$1
EV=${CELL_OUT:?}/ev
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/source.txt"
env CUDA_VISIBLE_DEVICES=0 cargo test --release -p memra-engine --offline --lib tier_transfer -- --include-ignored --test-threads=1
