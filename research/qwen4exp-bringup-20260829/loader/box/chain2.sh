#!/bin/bash
# Follow-on arm: re-run the STREAMING loader with the anon sampler live from t=0, so the new
# path's peak anon is a sampled maximum rather than a single mid-load reading. Waits for the
# old-cap230 arm to finish first; every arm takes the shared measurement lock inside run-cap.sh.
set -uo pipefail
q=/root/realgate/loaderout/CELL.log
say() { printf "[%s] %s\n" "$(date -u +%FT%TZ)" "$*" >> "$q"; }
while pgrep -f "loaderout/[r]un-cap.sh old-cap230" > /dev/null; do sleep 30; done
say "new-cap150b starting (streaming loader, sampler live from t=0)"
/root/realgate/loaderout/run-cap.sh new-cap150b /root/realgate/bin/qwen4exp_real_gate.loader 150G
say "new-cap150b rc=$?"
/root/realgate/loaderout/compare-identity.sh new-cap150b >> "$q" 2>&1
say "new-cap150b identity rc=$?"
say "CHAIN2 DONE"
