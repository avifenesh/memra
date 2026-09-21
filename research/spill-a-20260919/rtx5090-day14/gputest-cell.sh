#!/usr/bin/env bash
# The tier_transfer unit cells on the card, ignored ones included (the release test binary built by
# build.sh, --no-run first so this is a run, not a build), under the inherited collector lock.
# usage: gputest-cell.sh <lockfd>
set -uo pipefail
fd=$1
W=/home/avifenesh/projects/wt-spill-a
EV=${CELL_OUT:?}/ev
cd "$W"
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-5090.lock --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/source.txt"
env CUDA_VISIBLE_DEVICES=0 cargo test --release -p memra-engine --offline --lib tier_transfer -- --include-ignored --test-threads=1
