#!/usr/bin/env bash
# After the running cb33 plan (probes, bench) ends, keep the same lease: a validation run of the
# bf16 quality panel and a same-binary fp32 control panel (system python for the HumanEval
# sandbox), then the served ABAB cells; release.
set -u
LEASE=claude-code-20261107T032148-2828479
G=$HOME/.cache/memra-gdnbf16-real/gates
OUT=$HOME/.cache/memra-gdnbf16-real/runs/gb10-cb33-2
waitfor --gone 1241019 --timeout 600m --every 60s --quiet
QREP=2 QCLASS=1 CHAIN=gb10_chain2.sh bash "$G/gb10_plan.sh" cb33 "$LEASE" dev1 "$OUT/q-bf16-2" quality
QREP=1 QCLASS=0 CHAIN=gb10_chain2.sh bash "$G/gb10_plan.sh" cb33 "$LEASE" dev1 "$OUT/q-fp32-1" quality
CHAIN=gb10_chain2.sh bash "$G/gb10_plan.sh" cb33 "$LEASE" dev1 "$OUT/cells" cells
sparklock release "$LEASE" --unit cb33
echo "released $LEASE"
