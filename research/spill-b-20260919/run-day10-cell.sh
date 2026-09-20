#!/usr/bin/env bash
# Native single-PRO correctness only. All GPU execution belongs to the collector.
set -euo pipefail
cd /root/wt-b
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
label=${1:?label}; context=${2:?context}; case_name=${3:?case}; allocator=${4:-pooled}
root=/root/spill-receipts/b-day10/$label
mkdir -p "$root"
git rev-parse HEAD > "$root/source.commit"
sha256sum target/release/kv-tier-gate > "$root/binary.sha256"
args=(target/release/kv-tier-gate --artifact /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf --case "$case_name" --context "$context" --tiers host --same-program --kv-allocator "$allocator" --out "$root/receipt")
if [[ ${5:-} == diagnostic ]]; then args+=(--reclaim-diagnostic); fi
for attempt in {0..10}; do
  set +e
  python3 tools/tier-battery.py --rig pro-single --timeout 2400 --out "$root/collector-$attempt" --execute env NVIDIA_TF32_OVERRIDE=0 "${args[@]}" > "$root/collector-$attempt.log" 2>&1
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
  sleep 60
done
printf '1\n' > "$root/collector.exit"
exit 1
