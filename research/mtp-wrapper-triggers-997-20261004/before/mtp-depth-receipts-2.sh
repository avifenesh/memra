set -euo pipefail
python3 research/mtp-calibrated-depth-20260920/unpack_receipts.py --destination "$RUNNER_TEMP/mtp-cold"
python3 research/mtp-continuing-session-20260921/unpack_receipts.py --destination "$RUNNER_TEMP/mtp-warm"

