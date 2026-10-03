set -euo pipefail
python3 research/mtp-continuing-session-20260921/check_selection_rejections.py \
  --receipts "$RUNNER_TEMP/mtp-warm/experiment" --tmp "$RUNNER_TEMP" \
  --output "$RUNNER_TEMP/mtp-rejection-tests.json"
