#!/usr/bin/env bash
# WP-A day 32, B2 and B3 (DAY31 section 2): the day-26 double-park cell (pro-single-day26/double-park.sh, byte for byte) TWICE in
# ONE collector hold, first on the PRE-H2D binary (the log-only owner-segment field: B2's baseline, read in the same hold), then on
# the H2D binary. Twenty boots each, N=5 per arm per order, both orders. The first run's receipts sit under <R>/pre/double-park,
# the second's under <R>/double-park. Executed-not-qualified. usage: doublepark-pair.sh <lockfd>
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-day32
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
cd /root/wt-a || exit 1
mkdir -p "$R/pre"
rc=0
bash research/spill-a-20260919/pro-single-day26/double-park.sh "$fd" /root/wt-a "$R/pre" "$MODEL" "$R/bins/memra-server-pre" > "$R/pre/double-park.log" 2>&1 || rc=1
echo "$(date -u +%FT%TZ) pre double-park $(cat "$R/pre/double-park/ev/exit.txt" 2>/dev/null)" | tee -a "$R/progress.log"
bash research/spill-a-20260919/pro-single-day26/double-park.sh "$fd" /root/wt-a "$R" "$MODEL" "$R/bins/memra-server" > "$R/double-park.log" 2>&1 || rc=1
echo "$(date -u +%FT%TZ) h2d double-park $(cat "$R/double-park/ev/exit.txt" 2>/dev/null)" | tee -a "$R/progress.log"
exit $rc
