#!/bin/bash
# Build the DSpark drafting lane's binaries on the run box (src-spec -> target-spec), log to spec/build.log, end marker BUILD-RC.
# usage: spec_build.sh [check|build|clippy|fmt|test-server|test-engine]
set -uo pipefail
D=${LANE:?LANE names the working directory on the run box (checkpoint, oracle, dec ids, spec/, target-spec/)}
what=${1:-build}
export RUSTUP_HOME=$D/rust/rustup
export CARGO_HOME=$D/rust/cargo
export PATH=$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc
export MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-spec
export CARGO_BUILD_JOBS=16
export TMPDIR=$D/tmp
mkdir -p "$TMPDIR"
if [ ! -d "$CARGO_TARGET_DIR" ]; then
  cp -a $D/target-tp4 "$CARGO_TARGET_DIR"
fi
cd $D/src-spec
# rsync keeps the laptop mtimes; the first build here touched the lane's sources once (the target dir was seeded
# from another tree), every later edit is newer than its last build (the two clocks agree to a second)
c() { taskset -c 24-31,56-63 nice -n 5 cargo "$@"; }
case "$what" in
  check) c check --release -p memra-engine -p memra-server --bins --lib ;;
  build) c build --release -p memra-engine --bin dsv41-run --bin dsv41-gate --bin dsv41-bench --bin dsv41-spec \
           -p memra-server --bin memra-server ;;
  spec) c build --release -p memra-engine --bin dsv41-spec ;;
  check-spec) c check --release -p memra-engine --bin dsv41-spec --lib ;;
  cli) c build --release -p memra-cli --bin memra ;;
  clippy) c clippy --release --all-targets -p memra-engine -p memra-server -- -D warnings ;;
  clippy-mine) c clippy --release -p memra-engine --lib --bin dsv41-spec -- -D warnings \
                 && c clippy --release -p memra-server --lib -- -D warnings ;;
  fmt) c fmt --all -- --check ;;
  test-server) c test --release -p memra-server -j 16 --lib -- dsv41 ;;
  test-server-all) c test --release -p memra-server -j 16 --lib ;;
  test-engine) c test --release -p memra-engine -j 16 --lib -- dsv41 ;;
esac
echo "BUILD-RC $what $? $(date +%H:%M:%S)"
