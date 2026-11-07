#!/usr/bin/env bash
# After the running cb33 plan (quality, probes, bench) ends, keep the same lease and run the
# served ABAB cells there, then release.
set -u
LEASE=claude-code-20261107T032148-2828479
waitfor --gone 1241019 --timeout 600m --every 60s --quiet
bash "$HOME/.cache/memra-gdnbf16-real/gates/gb10_plan.sh" cb33 "$LEASE" dev1 "$HOME/.cache/memra-gdnbf16-real/runs/gb10-cb33" cells
sparklock release "$LEASE" --unit cb33
echo "released $LEASE"
