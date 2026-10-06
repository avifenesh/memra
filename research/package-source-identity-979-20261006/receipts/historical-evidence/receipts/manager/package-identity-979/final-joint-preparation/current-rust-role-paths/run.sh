#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /home/evidence-user/projects/evidence-source-worktree-979
[ "$(git rev-parse HEAD)" = b580822b28fbddfe32700198b67b03ba4e70fe46 ]
[ -z "$(git status --porcelain)" ]
printf '%s  %s\n' b4704f43a6261e7a90d1f60373914e6b05832b94c24919583edbe4f1fdd3c31b /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/INPUT-PINS.json | sha256sum --check
printf '%s  %s\n' f84b3ab355169d8f075b5097713ae004d8ffdcd899ac28bc67e358ada9c8f237 /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/observe-rustc.py | sha256sum --check
printf '%s  %s\n' e81a063e28e3fb8246a3ce54208da2425e0ef56f687e0492488a022072c1c3ac /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/witness.py | sha256sum --check
python3 -I /home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/final-joint-preparation/current-rust-role-paths/witness.py
