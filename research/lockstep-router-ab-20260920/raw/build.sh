#!/usr/bin/env bash
# Build base (origin/main) and fix (main + #565 lane commit) run-gen/run-lockstep, plus the CPU expert companion.
set -uo pipefail
source "$HOME/.cargo/env"
export PATH=/usr/local/cuda/bin:$PATH
cd /root/lane/memra && git checkout -q -B base origin/main
if [ ! -d /root/lane/memra-fix ]; then
  git worktree add -q -b fix /root/lane/memra-fix origin/main
  (cd /root/lane/memra-fix && git am -q /root/lane/0001-*.patch && git log --oneline -2)
fi
(cd /root/lane/memra && CARGO_TARGET_DIR=/root/lane/target-base cargo build --release -p memra-engine --bin run-gen --bin run_lockstep > /root/lane/build-base.log 2>&1 && echo BASE_OK >> /root/lane/build-base.log || echo BASE_FAIL >> /root/lane/build-base.log) &
(cd /root/lane/memra-fix && CARGO_TARGET_DIR=/root/lane/target-fix cargo build --release -p memra-engine --bin run-gen --bin run_lockstep > /root/lane/build-fix.log 2>&1 && echo FIX_OK >> /root/lane/build-fix.log || echo FIX_FAIL >> /root/lane/build-fix.log) &
(cd /root/lane/memra && bash tools/build_cpu_expert_companion.sh /root/lane/libmemra-cpu-experts.so > /root/lane/build-companion.log 2>&1 && echo COMPANION_OK >> /root/lane/build-companion.log || echo COMPANION_FAIL >> /root/lane/build-companion.log) &
wait
echo "BUILDS_DONE $(date -u +%T)"
