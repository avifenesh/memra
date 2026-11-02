#!/bin/bash
set -euo pipefail
printf '%s  %s\n' '84a2b23e5cbd4d83be8ebae1a47d35a831b4314c54571bb520e16523f34ac529' '/home/evidence-user/.local/state/evidence-campaign-20261002/receipts/A-package-flag-registry-1000/selective-handoff-root979-v3/run-self-output-control.py' | sha256sum --check --status
exec /usr/bin/python3 -I -B /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/A-package-flag-registry-1000/selective-handoff-root979-v3/run-self-output-control.py --work /home/evidence-user/.cache/memra-root979-self-output-control-v1 --records /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/self-output-control-v1
