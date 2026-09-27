#!/usr/bin/env bash
set -uo pipefail
cd /home/avifenesh/projects/wt-spill-d || exit 2
out=/tmp/spill-d-day14d
run() { local name=$1; shift; local start rc; start=$(date +%s); echo "== $name: $*"; "$@" > "$out/$name.log" 2>&1; rc=$?; echo "$rc" > "$out/$name.exit"; echo "== $name rc=$rc elapsed=$(( $(date +%s) - start ))s"; }
run fmt cargo fmt --all -- --check
run wrapper tools/portable-suites.sh
run teeth tools/test_portable_suites.sh
run gpu-ci-tests python3 tools/test_gpu_ci.py
run gpu-ci-floor tools/unittest-floor.sh tools test_gpu_ci.py 9
run pytest python3 -m pytest -q crates/memra-tier/tests/battery/
run check-flags bash tools/check-flags.sh
run docs-census bash tools/docs-registry-census.sh
run diff-check git diff --check
run yaml-load python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); yaml.safe_load(open('.github/workflows/gpu-ci.yml')); print('safe_load ok')"
run workflow-keys python3 tools/check-workflow-keys.py
run workflow-keys-teeth tools/test_workflow_keys.sh
run change-class-teeth tools/test_ci_change_class.sh
run unittest-floor-teeth tools/test_unittest_floor.sh
run action-pins tools/check-action-pins.sh
run shellcheck shellcheck -S warning tools/portable-suites.sh tools/ci-portable.sh tools/test_portable_suites.sh tools/test_workflow_keys.sh tools/hooks/pre-push
run hook-parse bash -n tools/hooks/pre-push
echo "== battery done"
