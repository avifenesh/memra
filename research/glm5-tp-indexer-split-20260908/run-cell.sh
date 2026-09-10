#!/bin/bash
set -euo pipefail
# name context split tc phase sampled check binary-class
name=$1
context=$2
split=$3
tc=$4
phase=${5:-none}
sampled=${6:-0}
check=${7:-0}
class=${8:-lane}
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
source /root/tunebox/env.sh
source research/glm5-tp-indexer-split-20260908/posture.sh
export MEMRA_GLM5_TP_INDEXER_SPLIT_PRIME=$split MEMRA_DSA_SCORE_TC=$tc
export MEMRA_GLM5_TP_INDEXER_SPLIT_CHECK=$check BOXP_SAMPLED=$sampled
export BOXP_PROFILE_PHASE=$phase
# Re-exec under the kernel lock. Merely finding the file never means it is held.
if [[ ${IDX_SPLIT_LOCKED:-0} != 1 ]]; then
  if [[ $check != 1 ]]; then
    while pgrep -x 'rustc|nvcc|ptxas|cicc' >/dev/null; do sleep 15; done
  fi
  exec flock "$MEMRA_GPU_LOCK" env IDX_SPLIT_LOCKED=1 bash "$0" "$@"
fi
out="$root/receipts/$name"
test ! -e "$out"
mkdir -p "$out"
bin="$root/target/release/glm5-tp2-box-probe"
if [[ $class == main ]]; then bin="$root/baseline-target/release/glm5-tp2-box-probe"; fi
date -u +%FT%TZ > "$out/start.utc"
sha256sum "$bin" > "$out/binary.sha256"
sha256sum "/root/tunebox/prompts/$context/long.txt" > "$out/prompt.sha256"
env | LC_ALL=C sort | grep -E '^(MEMRA_|BOXP_)' > "$out/env.txt"
nvidia-smi --query-gpu=index,name,uuid,power.limit,temperature.gpu,clocks.sm,memory.used --format=csv > "$out/gpu-before.csv"
ps -eo pid,ni,comm > "$out/processes-before.txt"
printf '%s START %s\n' "$(date -u +%FT%TZ)" "$name"
cmd=("$bin" /data/models/glm53-b200-mint "/root/tunebox/prompts/$context" "$out")
if [[ $phase != none ]]; then
  mkdir -p "$root/.lane/nsys"
  capture_end=stop
  if [[ $phase == both ]]; then capture_end=repeat:2; fi
  cmd=(nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end="$capture_end" --cuda-graph-trace=node --force-overwrite=true --output="$root/.lane/nsys/$name" "${cmd[@]}")
fi
set +e
/usr/bin/time -v "${cmd[@]}" > "$out/run.log" 2>&1
rc=$?
set -e
if [[ $rc == 0 ]]; then
  grep -F '[glm5-tp-sym-graph] engaged:' "$out/run.log" >/dev/null || rc=91
  if grep -E 'capture refused|capture-error|CUDA_ERROR|^Error:' "$out/run.log" >/dev/null; then rc=92; fi
  # The engagement predicate must match what the door actually prints. This door is PRIME-ONLY
  # (docs/FLAGS.md: decode ignores it and decode split dispatch was deleted) and its one
  # announcement is a once-per-process latch reading `t=<chunk> ... phase=prime`. The old
  # predicate grepped for `t=1 `, a decode shape this door can never emit, so every split=1 lane
  # row failed rc=93 while engaging correctly. Both halves are checked: ON must announce, and
  # OFF must NOT (the red half, without which the ON check proves only that grep works).
  if [[ $class == lane ]]; then
    if [[ $split == 1 ]]; then
      grep -E 'indexer=pool-split .*phase=prime' "$out/run.log" >/dev/null || rc=93
    else
      grep -F '[glm5-tp-indexer-split]' "$out/run.log" >/dev/null && rc=95
    fi
  fi
  if [[ $check == 1 ]]; then
    grep -F 'CHECK: merged idx plane byte-identical' "$out/run.log" >/dev/null || rc=94
  fi
fi
echo "$rc" > "$out/exit"
date -u +%FT%TZ > "$out/end.utc"
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv > "$out/gpu-after.csv"
printf '%s END %s rc=%s\n' "$(date -u +%FT%TZ)" "$name" "$rc"
tail -5 "$out/run.log"
exit "$rc"
