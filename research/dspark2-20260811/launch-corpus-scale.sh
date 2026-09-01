#!/usr/bin/env bash
# Launch corrected pilot ids 2,000..29,999 only after the 2K receipt and d2t freeze exist.

set -euo pipefail

workspace=/home/avifenesh/projects/wt-cx-dspark2
data_root=/data/projects/dspark/qwen35-9b-corpus-pilot
pilot_summary=$workspace/research/dspark2-20260811/raw/corpus-pilot-summary-02000.json
rank_root=$data_root/ranks/pilot-02000
remote_rank_root=/home/ubuntu/dspark2/artifacts/pilot-02000
range_log=$data_root/logs/range-pilot-scale.log
checkpoint_log=$data_root/logs/checkpoint-pilot-scale.log
launch_log=$workspace/research/dspark2-20260811/raw/corpus-pilot-scale-launch.log
range=$workspace/research/dspark2-20260811/run-corpus-range.sh
checkpoint=$workspace/research/dspark2-20260811/checkpoint-corpus-range.sh

mkdir -p "$data_root/logs" "$(dirname "$launch_log")"
exec > >(tee "$launch_log") 2>&1

echo "DSPARK-SCALE-LAUNCH-BEGIN $(date -Is)"
for required in "$pilot_summary" "$rank_root/sha256.txt" "$range" "$checkpoint"; do
  if [[ ! -f $required ]]; then
    echo "missing required 2K handoff file: $required" >&2
    exit 66
  fi
done
jq -e \
  '.format == "memra-dspark-corpus-summary-v2"
   and .label == "pilot"
   and .start == 0
   and .end == 2000
   and .pairs == 2000
   and (.assignment_cells | length) == 16' \
  "$pilot_summary" >/dev/null
(cd "$rank_root" && sha256sum -c sha256.txt)
ssh fpv-recognition-teacher bash -s -- "$remote_rank_root" <<'REMOTE_VERIFY'
set -euo pipefail
cd -- "$1"
sha256sum -c sha256.txt
REMOTE_VERIFY

if [[ -e $data_root/STOP-AFTER-CHUNK ]]; then
  echo "refusing scale launch while stop sentinel exists: $data_root/STOP-AFTER-CHUNK" >&2
  exit 65
fi
for unit in memra-dspark2-pilot.service memra-dspark2-pilot-checkpoint.service; do
  if systemctl --user is-active --quiet "$unit"; then
    echo "2K unit still active: $unit" >&2
    exit 75
  fi
done
for unit in memra-dspark2-pilot-scale.service memra-dspark2-pilot-scale-checkpoint.service; do
  if systemctl --user is-active --quiet "$unit"; then
    echo "scale unit already active: $unit" >&2
    exit 75
  fi
done

systemd-run --user \
  --unit memra-dspark2-pilot-scale \
  --collect \
  --property Nice=10 \
  --property IOSchedulingClass=best-effort \
  --property IOSchedulingPriority=7 \
  --property StandardOutput="append:$range_log" \
  --property StandardError="append:$range_log" \
  "$range" pilot 2000 30000 64
systemd-run --user \
  --unit memra-dspark2-pilot-scale-checkpoint \
  --collect \
  --property Nice=15 \
  --property StandardOutput="append:$checkpoint_log" \
  --property StandardError="append:$checkpoint_log" \
  "$checkpoint" pilot 2000 30000 64

systemctl --user show \
  memra-dspark2-pilot-scale.service \
  memra-dspark2-pilot-scale-checkpoint.service \
  -p Id -p ActiveState -p SubState -p InvocationID -p ExecStart
echo "DSPARK-SCALE-LAUNCH-DONE $(date -Is)"
