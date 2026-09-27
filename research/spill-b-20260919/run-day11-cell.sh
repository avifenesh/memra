#!/usr/bin/env bash
# Day 11 reclaim-cycle cell. Correctness probe only; all GPU execution belongs to the collector.
# usage: run-day11-cell.sh <rig pro-single|rtx5090> <label> <context> <cycles>
set -euo pipefail
rig=${1:?rig}; label=${2:?label}; context=${3:?context}; cycles=${4:?cycles}
case "$rig" in
  pro-single)
    wt=${WT:-/root/wt-b}
    artifact=${ARTIFACT:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
    root_dir=${ROOT:-/root/spill-receipts/b-day11}
    timeout=${TIMEOUT:-3600}
    export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
    ;;
  rtx5090)
    wt=${WT:-$HOME/projects/wt-spill-b}
    artifact=${ARTIFACT:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
    root_dir=${ROOT:-$wt/research/spill-b-20260919/rtx5090-day11}
    timeout=${TIMEOUT:-7200}
    ;;
  *) echo "REFUSED: unknown rig $rig" >&2; exit 2 ;;
esac
cd "$wt"
root=$root_dir/$label
mkdir -p "$root"
git rev-parse HEAD > "$root/source.commit"
sha256sum target/release/kv-tier-gate > "$root/binary.sha256"
args=(target/release/kv-tier-gate --artifact "$artifact" --case active --context "$context" --tiers host --same-program --kv-allocator vmm --reclaim-diagnostic --reclaim-cycles "$cycles" --out "$root/receipt")
for attempt in $(seq 0 "${ATTEMPTS:-10}"); do
  set +e
  python3 tools/tier-battery.py --rig "$rig" --timeout "$timeout" --out "$root/collector-$attempt" --execute env NVIDIA_TF32_OVERRIDE=0 "${args[@]}" > "$root/collector-$attempt.log" 2>&1
  rc=$?
  set -e
  if [[ $rc == 0 ]]; then
    printf '%s\n' "$attempt" > "$root/successful-attempt"
    printf '0\n' > "$root/collector.exit"
    exit 0
  fi
  # Retry lock contention only; preserve every refused attempt, never retry an executed failure.
  if [[ -e "$root/receipt" ]] || ! grep -q 'Resource temporarily unavailable' "$root/collector-$attempt.log"; then
    printf '%s\n' "$rc" > "$root/collector.exit"
    exit "$rc"
  fi
  sleep "${RETRY_SECONDS:-60}"
done
printf '1\n' > "$root/collector.exit"
exit 1
