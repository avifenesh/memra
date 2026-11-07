#!/usr/bin/env bash
# Rig: build the lane's 120a binaries (and the GPU test binaries) under the shared build lock,
# 2 jobs at nice 15, own target dir.  usage: rig-build.sh TAG
set -u
TAG=${1:?tag}
R=$HOME/.cache/memra-gdnbf16-decode
W=$R/wt
export CARGO_TARGET_DIR=$R/target-120a
L=$R/logs/build-$TAG.log
cd "$W" || exit 1
{
  echo "start $(date '+%F %T') HEAD=$(git rev-parse HEAD)"
  flock ~/.cache/memra-rig-build.lock nice -n 15 cargo build --release --locked -j 2 \
    -p memra-engine --bin kernel-check --bin run-gen --bin run-spec --bin decode-batch-bench \
    --bin argmax-margin-probe -p memra-server --bin memra-server
  rc=$?
  echo "BUILD=$rc"
  if [ $rc = 0 ]; then
    flock ~/.cache/memra-rig-build.lock nice -n 15 cargo test --release --locked -j 2 -p memra-engine \
      --test grid_capture_gpu --test gdn_packed_gpu --no-run
    echo "TESTBUILD=$?"
    mkdir -p "$R/bins/$TAG"
    for b in kernel-check run-gen run-spec decode-batch-bench argmax-margin-probe memra-server; do
      cp "$CARGO_TARGET_DIR/release/$b" "$R/bins/$TAG/"
    done
    sha256sum "$R/bins/$TAG"/* > "$R/bins/$TAG/sha256.txt"
  fi
  echo "RIG_BUILD_DONE $TAG $(date '+%F %T')"
} > "$L" 2>&1
