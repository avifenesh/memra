#!/usr/bin/env bash
# base = origin/main (one-job-per-row CPU program, the M=1 reference); lane = main + the exact multi-row program.
# The companion is built from the LANE tree (it exports memra_cpu_expert_rows_raw_v2); base never calls it.
set -uo pipefail
source "$HOME/.cargo/env"; export PATH=/usr/local/cuda/bin:$PATH
cd /root/lane/memra && git checkout -q -B base origin/main
if [ ! -d /root/lane/memra-lane ]; then git worktree add -q -b lane /root/lane/memra-lane origin/main && (cd /root/lane/memra-lane && git am -q /root/lane/0001-*.patch && git log --oneline -2); fi
(cd /root/lane/memra && CARGO_TARGET_DIR=/root/lane/target-base cargo build --release -p memra-engine --bin run-gen --bin run_lockstep > /root/lane/build-base.log 2>&1 && echo BASE_OK >> /root/lane/build-base.log || echo BASE_FAIL >> /root/lane/build-base.log) &
(cd /root/lane/memra-lane && CARGO_TARGET_DIR=/root/lane/target-lane cargo build --release -p memra-engine --bin run_lockstep --bin cpu_native_check > /root/lane/build-lane.log 2>&1 && echo LANE_OK >> /root/lane/build-lane.log || echo LANE_FAIL >> /root/lane/build-lane.log) &
(cd /root/lane/memra-lane && bash tools/build_cpu_expert_companion.sh /root/lane/libmemra-cpu-experts.so > /root/lane/build-companion.log 2>&1 && echo COMPANION_OK >> /root/lane/build-companion.log || echo COMPANION_FAIL >> /root/lane/build-companion.log) &
wait; echo "BUILDS_DONE $(date -u +%T)"
