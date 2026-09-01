#!/usr/bin/env bash
# stage-2/3 acceptance arm: boot NAME with env extras, sample, greedy+vendor 128-tok cells
# over both pools (card3-comparable shape: greedy = temperature 0, vendor = NO params).
# usage: acc_arm.sh NAME [--greedy-only] ENV=VAL...
set -uo pipefail
NAME="$1"; shift
GONLY=""
[ "${1:-}" = "--greedy-only" ] && { GONLY=1; shift; }
OUT=/root/out-specbat/acc/$NAME
/root/out-specbat/serve.sh start "acc-$NAME" "$@" || exit 1
python3 /root/out-specbat/run_pool.py sample --out "$OUT" || exit 1
python3 /root/out-specbat/run_pool.py cell --out "$OUT/greedy" --pool both --mode greedy --max-tokens 128
if [ -z "$GONLY" ]; then
  python3 /root/out-specbat/run_pool.py cell --out "$OUT/vendor" --pool both --mode vendor --max-tokens 128
fi
LOG=/root/out-specbat/logs/boot-acc-$NAME.log
echo "=== engagement: [glm5-acc] lines=$(grep -c "\[glm5-acc\]" "$LOG") ==="
grep "\[glm5-acc\]" "$LOG" | tail -2
grep -iE "spec-k|spec k=|choose_spec" "$LOG" | head -3 || true
