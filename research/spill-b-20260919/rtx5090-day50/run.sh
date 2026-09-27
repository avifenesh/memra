#!/usr/bin/env bash
# DAY50 stage 0 local half (the 5090, the 9B): concat-prime-probe from the lane checkout's release build (built at
# nice 19 under 600% before any card step, copied to target/day50 with its sha), then day50-stage0.sh at L = 6,144 and
# 30,720 (each run takes /tmp/memra-5090.lock with a bounded wait and yields 240 s after). Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919/rtx5090-day50
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/run.log"; }
if [ ! -x target/day50/concat-prime-probe ]; then
  echo "$(date -u +%FT%TZ) WP-B DAY50 probe build (nice 19, CPUQuota=600%)" >> "$WT/research/spill-b-20260919/cpu-concurrency.log"
  systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19 cargo build --release -p memra-engine \
    --bin concat-prime-probe > "$D/build.log" 2>&1
  rc=$?
  [ $rc = 0 ] || { log "build failed rc=$rc"; exit 2; }
  mkdir -p target/day50 && cp target/release/concat-prime-probe target/day50/
  git rev-parse HEAD > target/day50/source.commit
fi
sha256sum target/day50/concat-prime-probe > "$D/probe.sha256"; cp target/day50/source.commit "$D/probe.source"
log "start HEAD=$(git rev-parse HEAD) probe=$(cut -c1-16 "$D/probe.sha256") source=$(cat "$D/probe.source")"
LENGTHS=6144,30720 YIELD_S=240 bash research/spill-b-20260919/day50-stage0.sh "$D/stage0" "$WT/target/day50/concat-prime-probe" \
  "$MODEL" rtx5090
rc=$?
log "stage0 rc=$rc"
