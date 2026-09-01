#!/usr/bin/env bash
# Fresh release build for the fixed cx-requal source on the eu-west pair.
set -euo pipefail

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
ROOT=${REQUAL_ROOT:-/opt/scratch/nvme/cx-requal}
REPO=$ROOT/memra
EXPECTED_SOURCE=ac6ef049b8661008c0da91f4747f68f4dabdaa04
STAMP=${REQUAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${REQUAL_BUILD_OUT:-$ROOT/setup/build-$STAMP}

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
exec >"$OUT/build.log" 2>&1

echo "BUILD_START ts=$(date -u +%FT%TZ)"
source=$(git -C "$REPO" rev-parse HEAD)
echo "source=$source"
test "$source" = "$EXPECTED_SOURCE"
dirty=$(git -C "$REPO" status --porcelain --untracked-files=all)
test -z "$dirty" || { echo "$dirty"; echo "FAIL: source checkout is dirty"; exit 1; }
test ! -e "$REPO/target" || { echo "FAIL: release build is not fresh; target exists"; exit 1; }

hostname
uname -a
git -C "$REPO" log -5 --oneline --decorate
rustc --version
cargo --version
nvcc --version
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader

cd "$REPO"
nice -n 10 cargo build --release -p memra-engine \
    --bin kernel-check --bin run-gen --bin run-spec
nice -n 10 cargo build --release -p memra-server --bin memra-server

sha256sum \
    target/release/kernel-check \
    target/release/run-gen \
    target/release/run-spec \
    target/release/memra-server \
    >"$OUT/runtime-binaries.sha256"
git status --porcelain --untracked-files=all >"$OUT/git-status.txt"
test ! -s "$OUT/git-status.txt"
touch "$OUT/build.ok"
echo "BUILD_PASS ts=$(date -u +%FT%TZ)"
ln -sfn "$OUT" "$ROOT/setup/latest-build"
