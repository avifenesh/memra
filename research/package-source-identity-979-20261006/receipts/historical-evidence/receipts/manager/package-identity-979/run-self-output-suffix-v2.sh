#!/bin/bash
set -euo pipefail
printf '%s  %s\n' 'be99e84d69d0ac22501109b211ace3cebc14c285a326518645c2e500908bc99d' '/home/evidence-user/.local/state/evidence-campaign-20261002/receipts/A-package-flag-registry-1000/self-output-unfinished-suffix-v2/run-suffix.py' | sha256sum --check --status
exec /usr/bin/python3 -I -B /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/A-package-flag-registry-1000/self-output-unfinished-suffix-v2/run-suffix.py --work /home/evidence-user/.cache/memra-root979-self-output-control-v1 --records /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/self-output-control-suffix-v2
