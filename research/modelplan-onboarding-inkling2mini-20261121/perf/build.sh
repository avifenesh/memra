#!/bin/bash
# usage: build.sh [bins...]  (default ink2-gpu ink2-gpu-gate)
cd /home/avifenesh/.cache/inkling2mini/wt-perf
log=/home/avifenesh/.cache/inkling2mini/perf/build.log
bins="${@:-ink2-gpu ink2-gpu-gate}"
args=""; for b in $bins; do args="$args --bin $b"; done
waitfor --capacity 20,8 --timeout 10m && MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=/home/avifenesh/.cache/inkling2mini/target-perf systemd-run --user --scope -q -p CPUWeight=20 -p IOWeight=20 -- cargo build --release -p memra-engine $args > $log 2>&1
r=$?
echo "BUILD_RC=$r" >> $log
tail -3 $log
exit $r
