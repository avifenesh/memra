#!/usr/bin/env bash
set -uo pipefail
cd /home/avifenesh/projects/wt-spill-d || exit 2
out=/tmp/spill-d-day14/rerun; mkdir -p "$out"
run() { local name=$1; shift; local start rc; start=$(date +%s); echo "== $name: $*"; "$@" > "$out/$name.log" 2>&1; rc=$?; echo "$rc" > "$out/$name.exit"; echo "== $name rc=$rc elapsed=$(( $(date +%s) - start ))s"; }
run teeth tools/test_portable_suites.sh
run gpu-ci-tests python3 tools/test_gpu_ci.py
run gpu-ci-floor tools/unittest-floor.sh tools test_gpu_ci.py 9
run yaml-load python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); yaml.safe_load(open('.github/workflows/gpu-ci.yml')); print('safe_load ok')"
run workflow-keys python3 tools/check-workflow-keys.py
run workflow-keys-teeth tools/test_workflow_keys.sh
run change-class-teeth tools/test_ci_change_class.sh
run docs-census bash tools/docs-registry-census.sh
run diff-check git diff --check
echo "== rerun done"
