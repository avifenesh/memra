#!/usr/bin/env bash
# Day-14 local battery (CPU only, no GPU cell, DOCS_RS never set), one scope, sequential.
set -uo pipefail
cd /home/avifenesh/projects/wt-spill-d || exit 2
out=/tmp/spill-d-day14
run() {
    local name=$1; shift
    local start rc
    start=$(date +%s)
    echo "== $name: $*"
    "$@" > "$out/$name.log" 2>&1; rc=$?
    echo "$rc" > "$out/$name.exit"
    echo "== $name rc=$rc elapsed=$(( $(date +%s) - start ))s"
}
run fmt cargo fmt --all -- --check
run wrapper tools/portable-suites.sh
cp -f target/portable-suites.log "$out/wrapper-raw.log" 2>/dev/null
run teeth tools/test_portable_suites.sh
run gpu-ci-tests python3 tools/test_gpu_ci.py
run gpu-ci-floor tools/unittest-floor.sh tools test_gpu_ci.py 9
run pytest python3 -m pytest -q crates/memra-tier/tests/battery/
run unittest-lock-held bwrap --dev-bind / / --tmpfs /tmp --chdir "$PWD" bash -c '
    exec 8>/tmp/memra-5090.lock 7>/tmp/memra-gpu.lock
    flock -n 8 && flock -n 7 || { echo "could not take the private locks"; exit 3; }
    echo "holding /tmp/memra-5090.lock and /tmp/memra-gpu.lock in a private /tmp (bwrap) for the whole suite"
    tools/unittest-floor.sh crates/memra-tier/tests/battery "test_*.py" 80'
run check-flags bash tools/check-flags.sh
run docs-census bash tools/docs-registry-census.sh
run diff-check git diff --check
run yaml-load python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); yaml.safe_load(open('.github/workflows/gpu-ci.yml')); print('safe_load ok')"
run workflow-keys python3 tools/check-workflow-keys.py
run workflow-keys-teeth tools/test_workflow_keys.sh
run change-class-teeth tools/test_ci_change_class.sh
run unittest-floor-teeth tools/test_unittest_floor.sh
run action-pins tools/check-action-pins.sh
run action-pins-teeth tools/test_action_pins.sh
run shellcheck shellcheck -S warning tools/portable-suites.sh tools/ci-portable.sh tools/test_portable_suites.sh tools/test_workflow_keys.sh tools/hooks/pre-push
run bash-n bash -n tools/local-ci.sh
echo "== battery done"
