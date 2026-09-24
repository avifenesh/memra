#!/usr/bin/env bash
# DAY37 A2 (1.6, addendum B): the G1 grow series of on-demand planes, kv-tier-gate --case grow --kv-allocator
# vmm-ondemand at 32768 with the 27B artifact, through the collector (canonical rig lock + 250 ms telemetry).
# Correctness probe only. Retries lock contention only; an executed failure is never retried.
# usage: run-grow.sh <rig pro-single|rtx5090> <label>
set -euo pipefail
rig=${1:?rig}; label=${2:?label}
case "$rig" in
  pro-single)
    wt=${WT:-/root/wt-b}
    artifact=${ARTIFACT:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
    root_dir=${ROOT:-/root/spill-receipts/b-day37}
    timeout=${TIMEOUT:-7200}
    export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
    ;;
  rtx5090)
    wt=${WT:-$HOME/projects/wt-spill-b}
    artifact=${ARTIFACT:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
    root_dir=${ROOT:-$wt/research/spill-b-20260919/rtx5090-day37}
    timeout=${TIMEOUT:-10800}
    ;;
  *) echo "REFUSED: unknown rig $rig" >&2; exit 2 ;;
esac
cd "$wt"
bin=${GATE_BIN:-$wt/target/day37/kv-tier-gate}
root=$root_dir/$label
mkdir -p "$root"
git rev-parse HEAD > "$root/source.commit"
sha256sum "$bin" > "$root/binary.sha256"
args=("$bin" --artifact "$artifact" --case grow --context 32768 --tiers host --same-program --kv-allocator vmm-ondemand --out "$root/receipt")
for attempt in $(seq 0 "${ATTEMPTS:-60}"); do
  set +e
  python3 tools/tier-battery.py --rig "$rig" --timeout "$timeout" --out "$root/collector-$attempt" --execute env NVIDIA_TF32_OVERRIDE=0 "${args[@]}" > "$root/collector-$attempt.log" 2>&1
  rc=$?
  set -e
  if [[ $rc == 0 ]]; then
    printf '%s\n' "$attempt" > "$root/successful-attempt"
    printf '0\n' > "$root/collector.exit"
    exit 0
  fi
  if [[ -e "$root/receipt" ]] || ! grep -q 'Resource temporarily unavailable' "$root/collector-$attempt.log"; then
    printf '%s\n' "$rc" > "$root/collector.exit"
    exit "$rc"
  fi
  sleep "${RETRY_SECONDS:-120}"
done
printf '1\n' > "$root/collector.exit"
exit 1
