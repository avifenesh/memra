#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
proof=research/local-planning-io-972-20261003/tools
out=/home/avifenesh/.local/state/memra-rig-darklanes-20261002/receipts/C/local-planning-io-972/public-v3
mkdir -p "$out"
python3 "$proof/prove.py" "$PWD" "$out"
for helper in prove coherent_controls consumer_controls cli_controls; do
  if python3 -O "$proof/$helper.py" "$PWD" "$out" > "$out/$helper-O.log" 2>&1; then exit 1; fi
  /bin/grep -q 'assertions must be enabled' "$out/$helper-O.log"
done
tools/unittest-floor.sh tools 'test_validation_*.py' 185
tools/unittest-floor.sh tools test_check_support_states.py 21
python3 tools/validation_plan.py check-registry
git diff --check
