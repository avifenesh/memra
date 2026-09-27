#!/usr/bin/env bash
# DAY50 stage 1 local half (addendum D; the 5090, the 9B): the probe from this tree's release build (nice 19 under the
# 600% quota, copied to target/day50s1 with its source commit), then day50-stage1.sh at 6,144 and 30,720 under the hold
# runner on /tmp/memra-5090.lock. Never a signal to anything.
set -uo pipefail
WT=${WT:-$HOME/projects/wt-b-records}
D=$WT/research/spill-b-20260919/rtx5090-day50
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/stage1-run.log"; }
if [ ! -x target/day50s1/concat-prime-probe ]; then
  echo "$(date -u +%FT%TZ) WP-B DAY50 stage 1 probe build (nice 19, CPUQuota=600%, MemoryMax=12G)" >> "$WT/research/spill-b-20260919/cpu-concurrency.log"
  systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=12G nice -n 19 cargo build --release -p memra-engine \
    --bin concat-prime-probe > "$D/stage1-build.log" 2>&1
  rc=$?
  [ $rc = 0 ] || { log "build failed rc=$rc"; exit 2; }
  mkdir -p target/day50s1 && cp target/release/concat-prime-probe target/day50s1/
  git rev-parse HEAD > target/day50s1/source.commit
fi
sha256sum target/day50s1/concat-prime-probe > "$D/stage1-probe.sha256"; cp target/day50s1/source.commit "$D/stage1-probe.source"
log "start HEAD=$(git rev-parse HEAD) probe=$(cut -c1-16 "$D/stage1-probe.sha256") source=$(cat "$D/stage1-probe.source")"
WT=$WT RIG_LOCK=/tmp/memra-5090.lock CONTEXTS=6144,30720 LENGTHS=6144,30720 YIELD_S=240 \
  bash research/spill-b-20260919/day50-stage1.sh "$D/stage1" "$WT/target/day50s1/concat-prime-probe" "$MODEL" rtx5090
rc=$?
log "stage1 rc=$rc"
log "DAY50-STAGE1-LOCAL-DONE"
