#!/usr/bin/env bash
# Day 22 memra#427 cell F3 on the target card (run from /root/wt-b on the box): `kernel-check` on the FIX binary with
# the two manifests local-ci requires, through the collector with the inherited canonical lock (--rig pro-single,
# /tmp/memra-gpu.lock). The fix changes no kernel; the battery is owed by the brief. Pass/fail, not timed.
# usage: run-day22-kc-box3.sh <cell-name> <collector-timeout-s>
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}
R=${R:-/root/spill-receipts/b-day22}
KC=${KC:-/root/wt-b/target/release/kernel-check}
# The manifests to require; the 27b manifest's one cell (DUAL-BATCHED-AUX) needs the 9B artifact, which is not on
# the box, so a second cell may name the step35 manifest alone (MANIFESTS="tools/kernel-check-step35.cells").
MANIFESTS=${MANIFESTS:-"tools/kernel-check-27b.cells tools/kernel-check-step35.cells"}
MARGS=(); for m in $MANIFESTS; do MARGS+=(--require-manifest "$m"); done
cd /root/wt-b || exit 1
mkdir -p "$R"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
body='set -uo pipefail; K="$1"; O="$2"; shift 2; mkdir -p "$O"
MEMRA_KC_MODELS_DIR=/root/artifacts "$K" "$@" > "$O/kernel-check.log" 2>&1
echo "exit=$?" >> "$O/kernel-check.log"; grep "^SKIP \|^ALL GREEN\|^FAIL\|exit=" "$O/kernel-check.log"'
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" \
    --execute bash -c "$body" kc "$KC" "$out/cell" "${MARGS[@]}" > "$out-driver.log" 2>&1
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
