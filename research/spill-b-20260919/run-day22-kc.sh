#!/usr/bin/env bash
# Day 22 memra#427 cell F3, local RTX 5090: `kernel-check` on the FIX binary with the two manifests local-ci requires
# (`tools/kernel-check-27b.cells`, `tools/kernel-check-step35.cells`), under the collector. The fix changes no kernel;
# the battery is owed by the brief. Verdict line expected `ALL GREEN (N cells, K skipped)` with K within local-ci's
# budget of 11 on this rig. Bounded lock retries, never kills a holder.
# usage: run-day22-kc.sh <cell-name> <collector-timeout-s>
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}
WT=${WT:-$HOME/projects/wt-spill-b}
R=${R:-$WT/research/spill-b-20260919/rtx5090-day22}
KC=${KC:-$WT/target/release/kernel-check}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
cd "$WT" || exit 1
mkdir -p "$R"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
body='set -uo pipefail; K="$1"; O="$2"; mkdir -p "$O"
"$K" --require-manifest tools/kernel-check-27b.cells --require-manifest tools/kernel-check-step35.cells > "$O/kernel-check.log" 2>&1
echo "exit=$?" >> "$O/kernel-check.log"; grep "^SKIP \|^ALL GREEN\|^FAIL\|exit=" "$O/kernel-check.log"'
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  run python3 tools/tier-battery.py --rig rtx5090 --timeout "$tmo" --out "$out" \
    --execute bash -c "$body" kc "$KC" "$out/cell" > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && { git rev-parse HEAD > "$out/gate-source.txt"; sha256sum "$KC" > "$out/binary.sha256"; }
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-after.csv"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
