#!/usr/bin/env bash
# Polite on-box release rebuild for the pinned sigrouter-default profiling checkout.
set -euo pipefail

REPO=${NCUSPIKE_REPO:-/opt/scratch/nvme/memra-cx-ncuspike-src}
TARGET=${NCUSPIKE_TARGET:-/opt/scratch/nvme/memra-cx-ncuspike-target}
EXPECTED_SHA=1808220ead39d515a0854df49d1bb6452b558209
CARGO=${CARGO:-/home/ubuntu/.cargo/bin/cargo}

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
echo "build_start=$(date -u +%FT%TZ)"
echo "source_commit=$(git rev-parse HEAD)"
nice -n 15 env CARGO_TARGET_DIR="$TARGET" "$CARGO" build --release
echo "build_end=$(date -u +%FT%TZ)"
sha256sum "$TARGET/release/run-gen"
