#!/usr/bin/env bash
# Runs ON a GB10 unit under the caller's lease: build the lane tree (sm_121a) at low priority,
# then (WHAT=all, or WHAT=tests alone) the GPU test binaries, which build every engine bin;
# copy the binaries to bins/TAG.  usage: [WHAT=bins|tests|all] spark_build.sh TAG
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
R=$HOME/.cache/memra-gdnbf16-decode
TAG=${1:?tag}
export CARGO_TARGET_DIR=$R/target
mkdir -p "$R/logs" "$R/bins/$TAG/tests"
WHAT=${WHAT:-all}
L=$R/logs/build-$TAG-$WHAT.log
cd "$R/src" || exit 1
{
  echo "start $(date '+%F %T') src=$(cat "$R/src/.lane-head" 2>/dev/null)"
  rc=0
  if [ "$WHAT" != tests ]; then
    nice -n 10 ionice -c2 -n7 cargo build --release --locked -j ${JOBS:-6} \
      -p memra-engine --bin kernel-check --bin run-gen --bin run-spec --bin decode-batch-bench \
      --bin argmax-margin-probe -p memra-server --bin memra-server
    rc=$?
    echo "BUILD=$rc"
    if [ $rc = 0 ]; then
      for b in kernel-check run-gen run-spec decode-batch-bench argmax-margin-probe memra-server; do
        cp "$CARGO_TARGET_DIR/release/$b" "$R/bins/$TAG/"
      done
    fi
  fi
  if [ $rc = 0 ] && [ "$WHAT" != bins ]; then
    nice -n 10 ionice -c2 -n7 cargo test --release --locked -j ${JOBS:-6} -p memra-engine \
      --test grid_capture_gpu --test gdn_packed_gpu --no-run
    rc=$?
    echo "TESTBUILD=$rc"
    for t in grid_capture_gpu gdn_packed_gpu; do
      f=$(ls -t "$CARGO_TARGET_DIR"/release/deps/$t-* 2>/dev/null | grep -v '\.d$' | head -1)
      [ -n "$f" ] && cp "$f" "$R/bins/$TAG/tests/$t"
    done
    sha256sum "$R/bins/$TAG"/* "$R/bins/$TAG"/tests/* > "$R/bins/$TAG/sha256.txt" 2>/dev/null
  fi
  echo "SPARK_BUILD_DONE $TAG rc=$rc $(date '+%F %T')"
} > "$L" 2>&1
