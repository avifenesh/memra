#!/usr/bin/env bash
# The default arm through the engine-owned backing on this card: tier-transfer-gate conformance or
# roundtrip (every PASS line and byte_exact=true, as on the target card).
# usage: gate-cell.sh <lockfd> <conformance|roundtrip>
set -uo pipefail
fd=$1; case=$2
W=/home/avifenesh/projects/wt-spill-a
R=$W/research/spill-a-20260919/rtx5090-day14
EV=${CELL_OUT:?}/ev
cd "$W"
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-5090.lock --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/source.txt"
sha256sum "$R/bins/tier-transfer-gate" | tee "$EV/binary.sha256"
nvidia-smi --query-gpu=name,compute_cap,power.limit,driver_version --format=csv > "$EV/card.csv" 2>&1
env CUDA_VISIBLE_DEVICES=0 "$R/bins/tier-transfer-gate" "$case"
