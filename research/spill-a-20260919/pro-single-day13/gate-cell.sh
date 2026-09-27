#!/usr/bin/env bash
# The default arm through the engine-owned backing: tier-transfer-gate conformance or roundtrip
# (the refactor's regression check; every PASS line and byte_exact=true as on day 12).
# usage: gate-cell.sh <lockfd> <conformance|roundtrip>
set -uo pipefail
fd=$1; case=$2
R=/root/spill-receipts/a-day13
EV=${CELL_OUT:?}/ev
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/source.txt"
sha256sum "$R/bins/tier-transfer-gate" | tee "$EV/binary.sha256"
env CUDA_VISIBLE_DEVICES=0 "$R/bins/tier-transfer-gate" "$case"
