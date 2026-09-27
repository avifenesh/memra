#!/usr/bin/env bash
set -euo pipefail

study_root=${STUDY_ROOT:?set STUDY_ROOT to the pinned Qwen research directory}
cd "$study_root"
trap 'rc=$?; printf "exit=%s utc=%s\n" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > postscore-blind.exit; exit "$rc"' EXIT
grep -q '^exit=0 ' job-v2.exit

python3 -m py_compile \
    harness/confidence/heldout_workloads.py \
    harness/confidence/heldout_pair.py \
    harness/confidence/offline_adaptive.py \
    harness/confidence/report_fixed.py
python3 harness/confidence/verify_source.py \
    --base runtime-source.tar.gz \
    --patched runtime-source-confidence.tar.gz \
    --patch-receipt source-patch.json \
    --source-record source-confidence.json \
    --binary binaries-confidence/qwen-prefix-study \
    > source-verify.log 2>&1

STUDY_ROOT="$study_root" bash harness/confidence/sampled_gate.sh \
    > sampled-gate.log 2>&1
python3 harness/confidence/heldout_workloads.py \
    --binary binaries-confidence/qwen-prefix-study \
    --model models/qwen/target.gguf \
    --out heldout-workloads \
    > heldout-prep.log 2>&1
python3 harness/confidence/heldout_pair.py \
    --repo repo-confidence \
    --models models \
    --binaries binaries-confidence \
    --source source-confidence.json \
    --workloads heldout-workloads \
    --development-report fixed-grid-v2-report.json \
    --out heldout-pair \
    > heldout-pair.log 2>&1

python3 -m unittest discover -s harness/confidence \
    -p test_offline_adaptive.py -v \
    > offline-adaptive-test.log 2>&1
python3 harness/confidence/offline_adaptive.py \
    --root fixed-grid-v2 --out offline-adaptive.json \
    > offline-adaptive.log 2>&1
echo POSTSCORE_BLIND_COMPLETE
