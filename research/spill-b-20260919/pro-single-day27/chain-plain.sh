#!/usr/bin/env bash
# Day 27 extra pair, labelled, outside the pre-registered comparison: the same mix on the PLAIN path (MEMRA_SERVE_SPEC=0)
# so the park door has a plain pool to act on; default and compact, both arm orders, through the collector.
set -uo pipefail
R=/root/spill-receipts/b-day27
cd /root/wt-b || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> $R/chain.log; }
run_cell() {
  local name=$1 order=$2; shift 2
  local deadline=$((SECONDS + 1800))
  while ! flock -n /tmp/memra-gpu.lock true; do
    [ $SECONDS -ge $deadline ] && { log "$name: lock still held after 1800 s; not run"; return 3; }
    sleep 30
  done
  log "$name start"
  rm -rf $R/cell-$name
  python3 tools/tier-battery.py --rig pro-single --timeout 3600 --out $R/cell-$name --execute \
    env WT=/root/wt-b RIGDIR=$R/cells MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf MODEL_KEY=q38 LOCK=none NO_SCOPE=1 PORT=18526 MEMRA_SERVE_SPEC=0 "$@" \
    bash research/spill-b-20260919/run-day26-cell.sh $name $order > $R/collector-$name.log 2>&1
  log "$name exit=$? (cell exit $(cat $R/cells/$name.exit 2>/dev/null))"
  sleep 5
}
log "plain chain start HEAD=$(git rev-parse HEAD)"
run_cell plain-o1-default AB
run_cell plain-o1-compact AB MEMRA_KV_PARK_COMPACT=1
run_cell plain-o2-compact AB MEMRA_KV_PARK_COMPACT=1
run_cell plain-o2-default AB
log "plain chain done"
