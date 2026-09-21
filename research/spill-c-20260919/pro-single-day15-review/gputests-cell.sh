#!/usr/bin/env bash
# The two ignored GPU unit cells of the Option B unwind, on device 0 under the inherited collector lock.
# usage: gputests-cell.sh <lockfd>
set -uo pipefail
fd=$1
R=/root/spill-receipts/c-day15-review
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-c
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > $R/gputests/LOCK.json
git rev-parse HEAD
env CUDA_VISIBLE_DEVICES=0 cargo test --release -p memra-server --offline -- --ignored --test-threads=1 option_b_presubmit_refusal_returns_every_plane_and_keeps_the_tier_on option_b_postpublish_refusal_retires_the_ticket_and_keeps_the_tier_on
