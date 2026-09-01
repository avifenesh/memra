#!/usr/bin/env bash
# Incremental backup of box3 lane state to obj://darklanes-artifacts/ornith15/.
# The box is a SPOT instance (owner, 2026-08-20) — every poll re-runs this.
# Runs FROM the local rig: mints a fresh 1h STS session token and passes it via
# the ssh command environment only — credentials never touch box3's disk.
set -euo pipefail
# Box coordinates come from the environment (deployment facts live in darklanes).
KEY=${MEMRA_BOX3_KEY:?set MEMRA_BOX3_KEY}
BOX=${MEMRA_BOX3:?set MEMRA_BOX3 (user@host)}
BUCKET=obj://darklanes-artifacts/ornith15

CREDS=$(hyperscaler sts get-session-token --duration-seconds 3600 --output json | python3 -c "
import json, sys
c = json.load(sys.stdin)['Credentials']
print(f\"AWS_ACCESS_KEY_ID={c['AccessKeyId']} AWS_SECRET_ACCESS_KEY={c['SecretAccessKey']} AWS_SESSION_TOKEN={c['SessionToken']}\")
")

timeout 570 ssh -i "$KEY" "$BOX" "cd ~/models/ornith15 && \
  env $CREDS AWS_DEFAULT_REGION=Ohio hyperscaler s3 sync mtp-train/ $BUCKET/mtp-train/ --no-progress 2>&1 | tail -1 && \
  env $CREDS AWS_DEFAULT_REGION=Ohio hyperscaler s3 sync gates/ $BUCKET/gates/ --no-progress 2>&1 | tail -1 && \
  { [ -d st-gates ] && env $CREDS AWS_DEFAULT_REGION=Ohio hyperscaler s3 sync st-gates/ $BUCKET/st-gates/ --no-progress 2>&1 | tail -1 || true; } && \
  echo SYNCED"
