#!/usr/bin/env bash
# Build the measurement-only stage-owned-cache harness against the pinned clean runtime.
set -euo pipefail

REPO=${NCUSPIKE_REPO:-/opt/scratch/nvme/memra-cx-ncuspike-src}
HARNESS=${NCUSPIKE_HARNESS:-/opt/scratch/nvme/ncuspike-20260811/profile-harness}
TARGET=${NCUSPIKE_PROFILE_TARGET:-/opt/scratch/nvme/ncuspike-20260811/profile-target}
EXPECTED_SHA=1808220ead39d515a0854df49d1bb6452b558209
CARGO=${CARGO:-/home/ubuntu/.cargo/bin/cargo}

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain)"
echo "profile_build_start=$(date -u +%FT%TZ)"
echo "runtime_source_commit=$(git -C "$REPO" rev-parse HEAD)"
sha256sum "$HARNESS/Cargo.toml" "$HARNESS/src/main.rs"
nice -n 15 env CARGO_TARGET_DIR="$TARGET" "$CARGO" \
    build --release --manifest-path "$HARNESS/Cargo.toml"
echo "profile_build_end=$(date -u +%FT%TZ)"
sha256sum "$TARGET/release/ncuspike-profile"
test -z "$(git -C "$REPO" status --porcelain)"
