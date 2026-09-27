#!/usr/bin/env bash
# Day 23 copy for the MERGED tree (origin/main 653c997f4, #614 small_m_tier_max; the lane's scope removed): same seven arms, receipts under /root/spill-receipts/b-day23.
# Day 22 memra#427 cell F2 on the target card (run from /root/wt-b on the box): the continuation table on the FIX
# binary (prefill-rows scope; the gate counts a differing 16-row split as a failure), through the collector with the
# inherited canonical lock (--rig pro-single, /tmp/memra-gpu.lock). The local day-22 arms, byte for byte:
# 9296 (16..208), 9297 (17 49), 9311 (31 63), 9312 (32 64 96), the two chunk arms and the MEMRA_NO_BATCHED reference.
# MEMRA_PRIME_ROW_RECEIPT=1 is the existing per-call digest diagnostic. Pass/fail digests, not timed. Bounded lock
# retries, never kills a holder.
# usage: run-day23-cont-box3.sh <cell-name> <collector-timeout-s>
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}
R=${R:-/root/spill-receipts/b-day23}
G=${G:-/root/wt-b/target/release/qwen-a4-continuation-gate}
M=${M:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
P=${P:-/root/wt-b/docs/SERVING.md}
cd /root/wt-b || exit 1
mkdir -p "$R"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
body='set -uo pipefail; G="$1"; M="$2"; P="$3"; O="$4"; mkdir -p "$O"
arm() { name="$1"; shift; total="$1"; shift; tails="$1"; shift
  echo "== arm $name total=$total tails=$tails env: $*"
  env MEMRA_PRIME_ROW_RECEIPT=1 "$@" "$G" "$M" "$P" "$total" $tails > "$O/arm-$name.log" 2>&1
  echo "exit=$?" >> "$O/arm-$name.log"; grep -v "^\[prime-row\]" "$O/arm-$name.log"; }
arm F-9296 9296 "16 48 80 112 144 176 208"
arm F-9297 9297 "17 49"
arm F-9311 9311 "31 63"
arm F-9312 9312 "32 64 96"
arm F-9296-chunk32 9296 "16 48" MEMRA_PRIME_CHUNK=32
arm F-9296-chunk16 9296 "16 48" MEMRA_PRIME_CHUNK=16
arm F-9296-nobatched 9296 "16 48" MEMRA_NO_BATCHED=1'
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" \
    --execute bash -c "$body" fixgate "$G" "$M" "$P" "$out/cell" > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && { git rev-parse HEAD > "$out/gate-source.txt"; sha256sum "$G" "$M" > "$out/binary.sha256"; sha256sum "$P" > "$out/prompt.sha256"; }
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-after.csv"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
