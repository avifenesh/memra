#!/usr/bin/env bash
# A B B A as two pairs on the local RTX 5090, one gpulock lease per arm (`gpulock run --wait`), so the card can be
# yielded at every arm boundary. A pair whose two arms another lease came between (a gap over 90 s from the first
# arm's end to the second's start) keeps its receipts under a `.split` suffix and is run again.
#   pairs.sh WORKLOAD OUT [VAR=VALUE ...]     (CELL_A, CELL_B, LIMIT_ARGS as for cell.sh)
set -u
WL=$1; OUT=$2; shift 2
ENVARGS=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
log() { echo "$(TZ=Asia/Jerusalem date '+%F %T') $*" | tee -a "$OUT/pairs.log"; }
arm() { # index arm
  local i=$1 a=$2 bin=$CELL_A rc
  [ "$a" = B ] && bin=$CELL_B
  gpulock run --wait 4h --ttl 25m --for "memra #1124: 5090 above-floor $WL arm $i" -- \
    bash "$HERE/arm.sh" "$WL-$i-$a" "$bin" "$OUT" "$WL" "${ENVARGS[@]}" >> "$OUT/pairs.log" 2>&1
  rc=$?
  [ -f "$OUT/$WL-$i-$a/arm.txt" ] || { log "$WL $i $a did not run (gpulock rc=$rc); block stops"; exit 3; }
  log "$WL $i $a rc=$rc $(grep -a 'reclaim_on_defer_lines\|oom_lines' "$OUT/$WL-$i-$a/arm.txt" | tr '\n' ' ') $(grep -a -o 'budget [0-9]*MB' "$OUT/$WL-$i-$a/arm.txt" | head -1)"
}
log "pairs $WL A=$(basename "${CELL_A:?}") B=$(basename "${CELL_B:?}") env: ${ENVARGS[*]} LIMIT_ARGS=${LIMIT_ARGS:-}"
n=0
for pair in "A B" "B A"; do
  read -r first second <<< "$pair"
  while :; do
    n=$((n + 1)); i1=$n
    arm "$i1" "$first"
    t1=$(date +%s)
    n=$((n + 1)); i2=$n
    arm "$i2" "$second"
    t2=$(date -d "$(grep -a -o 'start=[^ ]*' "$OUT/$WL-$i2-$second/arm.txt" | head -1 | cut -d= -f2)" +%s)
    if [ $((t2 - t1)) -le 90 ]; then
      log "$WL pair $i1-$i2 ($first $second) intact: gap $((t2 - t1)) s"
      break
    fi
    log "$WL pair $i1-$i2 split by another lease (gap $((t2 - t1)) s): kept as .split, run again"
    mv "$OUT/$WL-$i1-$first" "$OUT/$WL-$i1-$first.split"
    mv "$OUT/$WL-$i2-$second" "$OUT/$WL-$i2-$second.split"
  done
done
log "pairs $WL done"
