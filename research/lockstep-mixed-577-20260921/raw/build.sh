#!/usr/bin/env bash
# Build the #577 probe tree (origin/main + the diagnostics commit): run-gen, run_lockstep, companion.
set -uo pipefail
source "$HOME/.cargo/env"; export PATH=/usr/local/cuda/bin:$PATH
cd /root/lane/memra && git checkout -q -B probe origin/main && git am -q /root/lane/0001-*.patch && git log --oneline -2
(cd /root/lane/memra && CARGO_TARGET_DIR=/root/lane/target cargo build --release -p memra-engine --bin run-gen --bin run_lockstep > /root/lane/build-probe.log 2>&1 && echo PROBE_OK >> /root/lane/build-probe.log || echo PROBE_FAIL >> /root/lane/build-probe.log) &
(cd /root/lane/memra && bash tools/build_cpu_expert_companion.sh /root/lane/libmemra-cpu-experts.so > /root/lane/build-companion.log 2>&1 && echo COMPANION_OK >> /root/lane/build-companion.log || echo COMPANION_FAIL >> /root/lane/build-companion.log) &
wait; echo "BUILDS_DONE $(date -u +%T)"
