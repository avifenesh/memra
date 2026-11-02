#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /home/evidence-user/projects/evidence-source-worktree-979
[ "$(git rev-parse HEAD)" = b580822b28fbddfe32700198b67b03ba4e70fe46 ]
[ -z "$(git status --porcelain)" ]
printf '%s  %s\n' a3c3f01622be9c4594026bc24efb99c900d0a04b78dbd0185271c3ce99747023 /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/INPUT-PINS.json | sha256sum --check
printf '%s  %s\n' 01819c2399dc9bafaada921723a6793793eb948022a13f5acc7241bdab7f768a /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/observe-rustc.py | sha256sum --check
printf '%s  %s\n' 0e0ae42adbefdbea538ab0f4573d85a71947f6ab8ca6e39791a75c850542bb5f /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/suffix-witness.py | sha256sum --check
python3 -I /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/suffix-witness.py
