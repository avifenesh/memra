#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /home/evidence-user/projects/evidence-source-worktree-979
[ "$(git rev-parse HEAD)" = b580822b28fbddfe32700198b67b03ba4e70fe46 ]
[ -z "$(git status --porcelain)" ]
printf '%s  %s\n' 93ee8643b5d6446129ddf58cfea28ed41f03bbd81f671e14f61491b83177aa76 /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/INPUT-PINS.json | sha256sum --check
printf '%s  %s\n' d1e2f36615a026a165080007c98cd5669678171cd6bfa1d64ccd13643bd3d27d /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/observe-rustc.py | sha256sum --check
printf '%s  %s\n' c9dd8af6c9dcb68df2344110472771dd3eab960e7452789fe03753924a144ccf /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/witness.py | sha256sum --check
python3 -I /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/witness.py
