#!/usr/bin/env bash
# FR-Spec trim battery, part 4 (memra#61): BRACKET the width knee.
#
# Cell A2's sweep is still rising at the widest ranks file it had (thinkon, tok/s:
# control 81.88 -> 8,192 86.26 -> 16,384 87.12 -> 32,768 88.20), which cannot tell a knee
# from a ceiling. The estimator puts the interior optimum just past 32,768 (1.052 there vs
# 1.048 at 49,152 and 1.043 at 65,536: the head saving shrinks with N faster than coverage
# improves). This sweeps 32,768 / 49,152 / 65,536 off the 65,536-wide blend so the shipped
# width is CHOSEN. Sweep only, one run per width plus the control — the claim stays the
# interleaved --trim-ab at the chosen width.
set -u
export CUDA_VISIBLE_DEVICES=0
export MEMRA_Q4E_SEAMS=idxsel,kvq,idxq
BIN=$HOME/realgate/bin/qwen4exp_real_gate.frspec2
CKPT=$HOME/data/q48fn-yarn1m
OUT=$HOME/realgate/frspec
SH=$HOME/realgate/shapes
OG64=$OUT/q4e-ranks-ogblend-65536.txt
LK=/tmp/q48fn-measure.lock
Q=$OUT/QUEUE.log
say(){ echo "[$(date -u +%FT%TZ)] frspec4: $*" >> "$Q"; }

say "fs2-og-thinkon-w64 acquiring -x"
flock -x "$LK" -c "$HOME/realgate/bin/q4e-load-lock.sh $OUT/F2-og-thinkon-w64.log $BIN $CKPT $OUT --label fs2-og-thinkon-w64 --prompts $SH/thinkon-prompts.tsv --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 --draft-trim $OG64 --draft-trim-n 65536 --trim-sweep 32768,49152,65536"
say "fs2-og-thinkon-w64 rc=$?"
