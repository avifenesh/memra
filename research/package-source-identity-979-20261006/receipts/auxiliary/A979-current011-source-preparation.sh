#!/usr/bin/env bash
set -euo pipefail
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export TMPDIR=/home/evidence-user/.cache/evidence-manager-20261006/hooks-scratch
for stage in capture-source-archives materialize-current-source prepare-current-registry-corpus; do
  timeout 65 systemd-run --user --wait --pipe --collect --unit="memra-rig-job-011-source-${stage}" --property=RuntimeMaxSec=60 --property=CPUQuota=200% --property=MemoryHigh=6G --property=MemoryMax=8G --property=MemorySwapMax=0 --property=TasksMax=256 --property=Nice=10 --property=Environment=TMPDIR=/home/evidence-user/.cache/evidence-manager-20261006/hooks-scratch --property=WorkingDirectory=/home/evidence-user/projects/evidence-source-worktree-979 /usr/bin/python3 -I -B "/home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/current-source-011a7709-preparation/${stage}.py"
done
