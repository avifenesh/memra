#!/usr/bin/env bash
# Fresh sm_120a release build for the GraphSession/B1FAST demotion-cost campaign.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"
ROOT=${GSCOST_ROOT:-/opt/dl-image/nvme/cx-gscost}
REPO=${GSCOST_REPO:-$ROOT/memra}
OUT=${GSCOST_BUILD_OUT:-$REPO/research/gscost-20260813/raw/build}
BUILD_TMP=${GSCOST_BUILD_TMP:-$ROOT/tmp}

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH

test -d "$REPO/.git" || { echo "FAIL: missing staged git checkout: $REPO"; exit 1; }
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SOURCE"
dirty=$(git -C "$REPO" status --porcelain --untracked-files=all)
test -z "$dirty" || { echo "$dirty"; echo "FAIL: staged source is dirty before build"; exit 1; }
test ! -e "$REPO/target" || { echo "FAIL: target exists; build would not be fresh"; exit 1; }
test ! -e "$OUT" || { echo "FAIL: build output already exists: $OUT"; exit 1; }

mkdir -p "$OUT" "$BUILD_TMP"
exec > >(tee "$OUT/build.log") 2>&1

echo "BUILD_START ts=$(date -u +%FT%TZ)"
echo "source=$EXPECTED_SOURCE"
echo "repo=$REPO"
echo "TMPDIR=$BUILD_TMP"
hostname
uname -a
git -C "$REPO" log -5 --oneline --decorate
rustc --version
cargo --version
nvcc --version
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader

cd "$REPO"
TMPDIR="$BUILD_TMP" nice -n 10 cargo build --release -p memra-engine \
    --bin kernel-check --bin run-gen --bin run-spec
TMPDIR="$BUILD_TMP" nice -n 10 cargo build --release -p memra-server --bin memra-server

sha256sum \
    target/release/kernel-check \
    target/release/run-gen \
    target/release/run-spec \
    target/release/memra-server \
    >"$OUT/runtime-binaries.sha256"
git status --porcelain --untracked-files=all >"$OUT/git-status-after.txt"
touch "$OUT/build.ok"
echo "BUILD_PASS ts=$(date -u +%FT%TZ)"
