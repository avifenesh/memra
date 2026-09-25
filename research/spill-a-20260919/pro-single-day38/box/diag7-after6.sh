#!/usr/bin/env bash
R=/root/spill-receipts/a-day38
until grep -q "diag6-cell rc=" $R/progress.log; do sleep 15; done
exec $R/diag7-run.sh
