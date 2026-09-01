#!/usr/bin/env bash
# Deterministic CPU-only SLRU policy analysis. No GPU, server, or live endpoint is used.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
LANE=$ROOT/research/slrutarget-20260813
RAW=$LANE/raw
MODEL=$LANE/traffic_model.lock.json

export TMPDIR=/home/avifenesh/tmp-lanes
mkdir -p "$TMPDIR"
test ! -e "$RAW" || { echo "FAIL: raw output already exists: $RAW"; exit 1; }
test ! -e "$LANE/analysis.json" || { echo "FAIL: analysis already exists"; exit 1; }
mkdir -p "$RAW"

{
    echo "timestamp=$(date -u +%FT%TZ)"
    echo "worktree=$ROOT"
    echo "branch=$(git branch --show-current)"
    echo "head=$(git rev-parse HEAD)"
    echo "base_tag=$(git describe --tags --exact-match v0.82.0^{} 2>/dev/null || echo v0.82.0)"
    echo "cachesize_source=$(jq -r .cachesize_source "$MODEL")"
    git cat-file -t "$(jq -r .cachesize_source "$MODEL")"
    git status --short --branch
    sha256sum "$MODEL" "$LANE/simulate.py" "$LANE/reduce.py" "$LANE/run-analysis.sh"
} 2>&1 | tee "$RAW/provenance.log"

python3 -m py_compile "$LANE/simulate.py" "$LANE/reduce.py" \
    2>&1 | tee "$RAW/python-compile.log"

python3 "$LANE/simulate.py" --model "$MODEL" \
    2>&1 | tee "$RAW/simulation.jsonl"

python3 "$LANE/reduce.py" "$RAW/simulation.jsonl" \
    2>&1 | tee "$LANE/analysis.json"

jq -e '.validation.verdict == "PASS"
    and .decision_checks.primary_nonnegative
    and .decision_checks.primary_scan_hits_zero
    and .decision_checks.primary_refusals_zero
    and .decision_checks.losing_shape_found' "$LANE/analysis.json" \
    2>&1 | tee "$RAW/reduction-gates.log"

jq '{
    primary,
    sensitivity_counts: {
        scenario_count: .sensitivity.scenario_count,
        slru_better_count: .sensitivity.slru_better_count,
        equal_count: .sensitivity.equal_count,
        slru_worse_count: .sensitivity.slru_worse_count
    },
    turnover: .controls.hotset_turnover_cycle,
    decision_checks
}' "$LANE/analysis.json" | tee "$RAW/reduction-summary.log"

find "$RAW" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum | tee "$RAW/SHA256SUMS"
