#!/usr/bin/env bash
# Day 13 final gate pair: base then fix, same gate source, same card, back to back.
set -uo pipefail
R=/root/spill-receipts/b-day13
bash $R/run-gate.sh main -final; echo "main-final rc=$?" >> $R/finals.log
bash $R/run-gate.sh fix -final; echo "fix-final rc=$?" >> $R/finals.log
