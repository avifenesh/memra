#!/usr/bin/env bash
# Runs ON a GB10 unit under the caller's lease: build the lane tree (sm_121a) at low priority and
# copy the binaries to bins/TAG. WHAT=server builds memra-server only (the census build);
# WHAT=bins adds the gate and bench binaries; WHAT=all adds the GPU test binaries.
#   [WHAT=server|bins|all] [JOBS=6] [SRC=dir] [TGT=target dir] spark_build.sh TAG
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
R=$HOME/.cache/memra-gdnfp16
TAG=${1:?tag}
export CARGO_TARGET_DIR=${TGT:-$R/target}
mkdir -p "$R/logs" "$R/bins/$TAG/tests"
WHAT=${WHAT:-bins}
L=$R/logs/build-$TAG-$WHAT.log
SRC=${SRC:-$R/src}
cd "$SRC" || exit 1
{
  echo "start $(date '+%F %T') src=$(cat "$SRC/.lane-head" 2>/dev/null) what=$WHAT"
  bins="memra-server"
  args=(-p memra-server --bin memra-server)
  if [ "$WHAT" != server ]; then
    bins="kernel-check run-gen run-spec decode-batch-bench memra-server"
    args=(-p memra-engine --bin kernel-check --bin run-gen --bin run-spec --bin decode-batch-bench
      -p memra-server --bin memra-server)
  fi
  nice -n 10 ionice -c2 -n7 cargo build --release --locked -j "${JOBS:-6}" "${args[@]}"
  rc=$?
  echo "BUILD=$rc"
  if [ $rc = 0 ]; then
    for b in $bins; do cp "$CARGO_TARGET_DIR/release/$b" "$R/bins/$TAG/"; done
  fi
  if [ $rc = 0 ] && [ "$WHAT" = all ]; then
    nice -n 10 ionice -c2 -n7 cargo test --release --locked -j "${JOBS:-6}" -p memra-engine \
      --test grid_capture_gpu --test gdn_packed_gpu --no-run
    rc=$?
    echo "TESTBUILD=$rc"
    for t in grid_capture_gpu gdn_packed_gpu; do
      f=$(ls -t "$CARGO_TARGET_DIR"/release/deps/$t-* 2>/dev/null | grep -v '\.d$' | head -1)
      [ -n "$f" ] && cp "$f" "$R/bins/$TAG/tests/$t"
    done
  fi
  sha256sum "$R/bins/$TAG"/* "$R/bins/$TAG"/tests/* > "$R/bins/$TAG/sha256.txt" 2>/dev/null
  echo "SPARK_BUILD_DONE $TAG rc=$rc $(date '+%F %T')"
} > "$L" 2>&1
