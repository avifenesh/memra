#!/usr/bin/env bash
# stage-1 spec arm: boot spec K=$1 (optionally FRSPEC_TRIM=$2), sample, greedy tapes, compare vs plain.
set -uo pipefail
K="$1"; TRIM="${2:-}"
NAME="s1-k$K"; EXTRA=()
[ -n "$TRIM" ] && { NAME="s1-k$K-trim"; EXTRA=(MEMRA_FRSPEC_TRIM="$TRIM"); }
OUT=/root/out-specbat/s1/$NAME
/root/out-specbat/serve.sh start "$NAME" MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_K="$K" "${EXTRA[@]}" || exit 1
python3 /root/out-specbat/run_pool.py sample --out "$OUT" || exit 1
python3 /root/out-specbat/run_pool.py cell --out "$OUT" --pool both --mode greedy --max-tokens 256
echo "=== IDENTITY $NAME vs plain ==="
python3 /root/out-specbat/run_pool.py compare --a /root/out-specbat/s1/plain --b "$OUT"
rc=$?
echo "=== acc lines (tail) ==="
grep -c "\[glm5-acc\]" /root/out-specbat/logs/boot-$NAME.log
grep "\[glm5-acc\]" /root/out-specbat/logs/boot-$NAME.log | tail -3
exit $rc
