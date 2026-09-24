#!/usr/bin/env bash
# WP-A days 33 and 34 on BOX4 (DAY34.md section 4): the day-26 double-park cell (pro-single-day26/double-park.sh, byte for
# byte) three times in ONE collector hold: the day-32 H2D binary (<R>/d32), the day-33 binary (<R>/d33), the day-34
# binary (<R>, the tip). Twenty boots each, N=5 per arm per order, both orders. Executed-not-qualified.
# usage: doubleparks.sh <lockfd>
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-day34
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
cd /root/wt-a || exit 1
rc=0
for name in d32 d33; do
  mkdir -p "$R/$name"
  bash research/spill-a-20260919/pro-single-day26/double-park.sh "$fd" /root/wt-a "$R/$name" "$MODEL" "$R/bins/memra-server-$name" > "$R/$name/double-park.log" 2>&1 || rc=1
  echo "$(date -u +%FT%TZ) $name double-park $(cat "$R/$name/double-park/ev/exit.txt" 2>/dev/null)" | tee -a "$R/progress.log"
done
bash research/spill-a-20260919/pro-single-day26/double-park.sh "$fd" /root/wt-a "$R" "$MODEL" "$R/bins/memra-server-d34" > "$R/double-park.log" 2>&1 || rc=1
echo "$(date -u +%FT%TZ) d34 double-park $(cat "$R/double-park/ev/exit.txt" 2>/dev/null)" | tee -a "$R/progress.log"
exit $rc
