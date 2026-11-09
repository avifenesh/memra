#!/bin/bash
# Build the lane's binaries and the suites the change touches (no GPU), then run the suites inside a short one-card lease
# (some engine and server tests create a CUDA context): release bins, clippy -D warnings on the touched crates,
# memra-server and memra-cli lib suites, the engine's dsv41 unit tests, fmt.
# usage: test_serve.sh [build|run|all]
. "${LANE_ENV:?LANE_ENV names the machine file that sets ROOT GPU_UUID BOX_LOCK QUIET_FILE CORES GPU_RUN BUILD}"
set -u
D=$ROOT
B=$BUILD
L=$D/serve/logs
UUID=$GPU_UUID
what=${1:-all}
mkdir -p $L
step() {
  local name=$1; shift
  "$@" > $L/$name.log 2>&1
  local rc=$?
  echo "$name rc $rc $(grep -E 'test result:|Finished|^error' $L/$name.log | tail -n 3 | tr '\n' ' ')"
}
if [ "$what" = build ] || [ "$what" = all ]; then
  step build-bins $B build --release -p memra-engine --bin dsv41-run -p memra-server --bin memra-server -p memra-cli --bin memra
  step clippy $B clippy --release --all-targets -p memra-engine -p memra-server -p memra-cli -- -D warnings
  step build-tests $B test --release -p memra-server -p memra-cli -p memra-engine --lib --no-run
  step fmt $B fmt --all -- --check
fi
if [ "$what" = run ] || [ "$what" = all ]; then
  if [ -e $QUIET_FILE ]; then echo "QUIET is set: not running the suites"; exit 3; fi
  if [ -e $D/serve/STOP ]; then echo "STOP is set: not running the suites"; exit 4; fi
  R=$D/leases/serve-tests-$(date +%Y%m%dT%H%M%S); mkdir -p "$R"
  $GPU_RUN --gpus $UUID --receipt "$R" --box-lock $BOX_LOCK --timeout 2400 --wait 1800 -- \
    bash -c "$B test --release -p memra-server --lib > $L/test-server.log 2>&1; echo server \$?; \
             $B test --release -p memra-cli --lib > $L/test-cli.log 2>&1; echo cli \$?; \
             $B test --release -p memra-engine --lib dsv41 > $L/test-engine-dsv41.log 2>&1; echo engine-dsv41 \$?"
  for f in test-server test-cli test-engine-dsv41; do echo "$f: $(grep -E 'test result:' $L/$f.log | tail -n 1)"; done
fi
echo TEST-SERVE-DONE
