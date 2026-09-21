#!/usr/bin/env bash
# The ignored GPU unit cells of both contract routes (Option B's two, Option C's six) on device 0
# under the inherited collector lock. usage: gputests-cell.sh <lockfd>
set -uo pipefail
fd=$1
R=/root/spill-receipts/c-day16-review
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-c
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > $R/gputests/LOCK.json
git rev-parse HEAD
env CUDA_VISIBLE_DEVICES=0 cargo test --release -p memra-server --offline -- --ignored --test-threads=1 \
  option_b_presubmit_refusal_returns_every_plane_and_keeps_the_tier_on \
  option_b_postpublish_refusal_retires_the_ticket_and_keeps_the_tier_on \
  option_c_promote_routes_every_contract_plane_and_keeps_the_host_twin \
  option_c_presubmit_refusal_releases_every_destination_and_keeps_the_host_twin \
  option_c_postpublish_refusal_retires_the_ticket_and_keeps_the_host_twin \
  option_c_receipt_mismatch_cancels_before_publication_and_recovers_the_source \
  option_c_partial_acceptance_unwinds_refused_with_every_destination_released \
  option_c_first_ready_view_failure_unwinds_through_the_published_arm
