#!/bin/bash
set -euo pipefail
coord=/home/avifenesh/.local/state/memra-rig-20261002
python3 "$coord/scratch/A/review-lifecycle-retry.py" --promtool "$coord/receipts/A/tools/promtool" --credentials-dir "$coord/scratch/A/retry-credentials"
