#!/usr/bin/env bash
# Day 23 task 1 on the target card (run from /root/wt-b on the box): `kernel-check` on the fixed tree (lane tip
# aeb4d0771; no `crates/` file differs from the day-22 fix build, binary SHA-256 d61a41b5...) through the collector
# with the canonical lock (--rig pro-single, /tmp/memra-gpu.lock). Two cells, pass/fail, not timed:
#   kc-step35  MANIFESTS="tools/kernel-check-step35.cells"                      the step35 manifest alone
#   kc-full    MANIFESTS="tools/kernel-check-27b.cells tools/kernel-check-step35.cells"
#              with the 9B artifact the 27b manifest's DUAL-BATCHED-AUX cell needs, staged from the verified local
#              copy (/data on the dev rig, SHA-256 52c9cceb... equal to five receipts in the repo) into this lane's
#              own receipts dir; MODELS_DIR is a directory of references (symlinks) to the two /root/artifacts files
#              plus the staged 9B. /root/artifacts is not touched. Neither the manifests nor the checker are edited.
# usage: run-day23-kc-box3.sh <cell-name> <collector-timeout-s>
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}
R=${R:-/root/spill-receipts/b-day23}
KC=${KC:-/root/wt-b/target/release/kernel-check}
MODELS_DIR=${MODELS_DIR:-/root/spill-receipts/b-day23/models}
MANIFESTS=${MANIFESTS:-"tools/kernel-check-27b.cells tools/kernel-check-step35.cells"}
MARGS=(); for m in $MANIFESTS; do MARGS+=(--require-manifest "$m"); done
cd /root/wt-b || exit 1
mkdir -p "$R"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
body='set -uo pipefail; K="$1"; O="$2"; MD="$3"; shift 3; mkdir -p "$O"
MEMRA_KC_MODELS_DIR="$MD" "$K" "$@" > "$O/kernel-check.log" 2>&1
echo "exit=$?" >> "$O/kernel-check.log"; grep "^SKIP \|^ALL GREEN\|^FAIL\|MISSING\|required cell\|exit=" "$O/kernel-check.log"'
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" \
    --execute bash -c "$body" kc "$KC" "$out/cell" "$MODELS_DIR" "${MARGS[@]}" > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && { git rev-parse HEAD > "$out/gate-source.txt"; sha256sum "$KC" > "$out/binary.sha256"; echo "$MANIFESTS" > "$out/manifests.txt"; ls -la "$MODELS_DIR" > "$out/models-dir.txt"; }
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-after.csv"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
