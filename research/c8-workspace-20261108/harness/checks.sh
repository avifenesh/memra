#!/usr/bin/env bash
# The lane's local CPU checks on the lane worktree, under the rig build lock, in a target of their
# own: fmt, clippy -D warnings on the touched packages, the memra-server and memra-engine lib tests.
set -o pipefail
WT=<tree>
cd "$WT" || exit 2
export CARGO_TARGET_DIR=<lane>/target-test-lane MEMRA_CUDA_ARCH=120a
L=~/.cache/memra-rig-build.lock
O=<lane>/logs/checks-$(git rev-parse --short=10 HEAD)
mkdir -p "$O"
echo "checks $(git rev-parse --short=10 HEAD) $(date -Is)"
cargo fmt --all -- --check > "$O/fmt.log" 2>&1; echo "fmt rc=$?"
git diff --check origin/main HEAD > "$O/diff-check.log" 2>&1; echo "diff-check rc=$?"
flock "$L" nice -n 15 cargo clippy -j 2 -p memra-engine -p memra-server --all-targets -- -D warnings > "$O/clippy.log" 2>&1
echo "clippy rc=$? $(grep -c '^error' "$O/clippy.log") errors"
flock "$L" nice -n 15 cargo test -j 2 -p memra-server --lib > "$O/server-lib.log" 2>&1
echo "server-lib rc=$? $(grep -E '^test result' "$O/server-lib.log" | tail -n 1)"
flock "$L" nice -n 15 cargo test -j 2 -p memra-engine --lib > "$O/engine-lib.log" 2>&1
echo "engine-lib rc=$? $(grep -E '^test result' "$O/engine-lib.log" | tail -n 1)"
( cd tools && for t in test_local_build_cache test_release_input_view test_release_inputs test_release_qualification test_cell_gate test_validation_plan; do
    env -u CARGO_TARGET_DIR timeout 1200 nice -n 10 python3 -m unittest $t > "$O/$t.log" 2>&1; echo "$t rc=$? $(tail -n 1 "$O/$t.log")"; done )
echo "CHECKS_DONE $(date -Is)"
