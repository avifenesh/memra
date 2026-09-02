#!/usr/bin/env bash
# FR-Spec draft-head trim battery at corpus scale (memra#61, lane/q4e-frspec-trim-20260902).
#
# Card 0 only (cards 2-3 belong to the #53 lane; card 1 is another lane's). Serving caches
# (MEMRA_Q4E_SEAMS=idxsel,kvq,idxq; selgroup is default-ON since PR #56 and is NOT pinned
# here). Two locks, both mandatory: the box LOAD serializer (two concurrent 174 GB loads
# OOM the host) and the measurement lock around every invocation that quotes a number.
#
# Cells, in priority order (the serving shape first):
#   A  thinkon at the mtp10 ship policy (adapt k_lo=1 + pmin 0.3), ogblend ranks:
#      width sweep -> hidden/greedy goldens -> sampled twin on BOTH arms -> spec-gate byte
#      identity at 256 tokens -> the interleaved trim A/B (x3, arm order flipped per rep).
#   B  raw fixed K=5, ogblend ranks — the shape mtp10 measured at 0.8824 (bench-only:
#      short friendly prompt, greedy-loop law applies to any degenerate continuation).
#   C  raw fixed K=5, PURE-CORPUS (sxc) ranks — the estimator predicts 0.82 here against
#      1.02 for ogblend, so this cell is what makes the free estimator a receipt rather
#      than a story: same width, same shape, only the corpus source differs.
#   D  thinkoff at ship policy, ogblend ranks — the third served shape.
set -u
export CUDA_VISIBLE_DEVICES=0
export MEMRA_Q4E_SEAMS=idxsel,kvq,idxq
BIN=$HOME/realgate/bin/qwen4exp_real_gate.frspec
CKPT=$HOME/data/q48fn-yarn1m
OUT=$HOME/realgate/frspec
SH=$HOME/realgate/shapes
OG=$OUT/q4e-ranks-ogblend-32768.txt
SXC=$OUT/q4e-ranks-sxc32768.txt
LK=/tmp/q48fn-measure.lock
Q=$OUT/QUEUE.log
mkdir -p "$OUT"
say(){ echo "[$(date -u +%FT%TZ)] frspec: $*" >> "$Q"; }

cell(){ # cell <label> <logfile> <args...>
  local label=$1 log=$2; shift 2
  say "$label acquiring -x"
  flock -x "$LK" -c "$HOME/realgate/bin/q4e-load-lock.sh $OUT/$log $BIN $CKPT $OUT --label $label $*"
  say "$label rc=$?"
}

case "${1:-all}" in
A|all)
  cell fs-og-thinkon A-og-thinkon.log \
    --prompts "$SH/thinkon-prompts.tsv" --goldens "$HOME/realgate/dump" \
    --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 \
    --draft-trim "$OG" --draft-trim-n 32768 --trim-sweep 8192,16384,32768 \
    --spec-sampled --spec-gate 256 --trim-ab 3x256
  ;;& # fall through when "all"
B|all)
  cell fs-og-raw B-og-raw.log \
    --prompts "$HOME/realgate/dump/prompts.tsv" \
    --spec-k 5 \
    --draft-trim "$OG" --draft-trim-n 32768 \
    --spec-sampled --spec-gate 256 --trim-ab 3x256
  ;;&
C|all)
  cell fs-sxc-raw C-sxc-raw.log \
    --prompts "$HOME/realgate/dump/prompts.tsv" \
    --spec-k 5 \
    --draft-trim "$SXC" --draft-trim-n 32768 \
    --spec-gate 256 --trim-ab 3x256
  ;;&
D|all)
  cell fs-og-thinkoff D-og-thinkoff.log \
    --prompts "$SH/thinkoff-prompts.tsv" \
    --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 \
    --draft-trim "$OG" --draft-trim-n 32768 \
    --spec-sampled --spec-gate 256 --trim-ab 3x256
  ;;&
E|all)
  # LAW:interleave-x3-default escalation, rule (a): cell A's full arm reported
  # spread_pct=0.510 (>0.5%), so the affected PAIR is extended to x5, still interleaved,
  # same boot shape, arm order still flipped per rep.
  cell fs-og-thinkon-x5 E-og-thinkon-x5.log \
    --prompts "$SH/thinkon-prompts.tsv" \
    --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 \
    --draft-trim "$OG" --draft-trim-n 32768 \
    --spec-sampled --trim-ab 5x256
  ;;
esac
say "done ${1:-all}"
