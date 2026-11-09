#!/usr/bin/env bash
# The lane's unit tests (engine shape arithmetic, the admission charge, the batch OOM card key, the
# responses test) under the rig build lock, in a test target of their own.
#   unit.sh WORKTREE [extra filters for memra-server]
set -o pipefail
WT=${1:?worktree}; shift
cd "$WT" || exit 2
export CARGO_TARGET_DIR=<lane>/target-test MEMRA_CUDA_ARCH=120a
L=~/.cache/memra-rig-build.lock
echo "unit $(git rev-parse --short=10 HEAD) $(date -Is)"
flock "$L" nice -n 15 cargo test -j 2 -p memra-engine --lib -- prime_workspace_shape plain_prime_holds 2>&1 | grep -E '^test |test result|error|panicked|FAILED' | tail -n 20
e=${PIPESTATUS[0]}
flock "$L" nice -n 15 cargo test -j 2 -p memra-server --lib -- plain_admission predictive_charge restore batch_oom_recovery background_with_stream "$@" 2>&1 | grep -E '^test |test result|error|panicked|FAILED' | tail -n 60
s=${PIPESTATUS[0]}
echo "UNIT_DONE engine=$e server=$s $(date -Is)"
