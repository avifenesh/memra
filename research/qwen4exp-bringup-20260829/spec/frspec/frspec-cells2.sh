#!/usr/bin/env bash
# FR-Spec trim battery, RE-RUN on the shipped program (memra#61).
#
# Why a re-run: the first pass (box-pre57/) was cut from a branch based on 24d775458, before
# PR #57's StepPool shed landed on main. Both arms of an A/B always share one binary, so
# those ratios are internally valid — but a trim's payoff is a SHARE of the round, and a
# change that moves round composition can move the share. The claim rows must therefore be
# cut from a binary that carries the shipped program (rebased onto c04c1da9b: #57 shed,
# #56 selgroup default-ON, v0.124.0).
#
# x5 everywhere, not x3: EVERY arm in the first pass reported a within-arm spread above the
# 0.5% escalation threshold of LAW:interleave-x3-default (thinkon 0.51/0.44, raw 1.77/3.04,
# sxc-raw 2.09/2.23, thinkoff 1.83/3.44), and the raw verdict (1.0026) sat well inside the
# pooled spread — rules (a) and (b) both fired, so the pairs are extended, still interleaved,
# with the arm order still flipped on odd reps.
#
# A NOTE ON DRIVING THIS: do not edit a cell script while bash is executing it. The first
# pass lost its last cell to exactly that — appending a case arm shifted bash's read offset
# and it died with `syntax error near unexpected token )` after cell D.
set -u
export CUDA_VISIBLE_DEVICES=0
export MEMRA_Q4E_SEAMS=idxsel,kvq,idxq
BIN=$HOME/realgate/bin/qwen4exp_real_gate.frspec2
CKPT=$HOME/data/q48fn-yarn1m
OUT=$HOME/realgate/frspec
SH=$HOME/realgate/shapes
OG=$OUT/q4e-ranks-ogblend-32768.txt
SXC=$OUT/q4e-ranks-sxc32768.txt
LK=/tmp/q48fn-measure.lock
Q=$OUT/QUEUE.log
mkdir -p "$OUT"
say(){ echo "[$(date -u +%FT%TZ)] frspec2: $*" >> "$Q"; }

cell(){ # cell <label> <logfile> <args...>
  local label=$1 log=$2; shift 2
  say "$label acquiring -x"
  flock -x "$LK" -c "$HOME/realgate/bin/q4e-load-lock.sh $OUT/$log $BIN $CKPT $OUT --label $label $*"
  say "$label rc=$?"
}

# A: the serving shape, ship policy. Carries the width sweep, the goldens/hidden gates, the
#    vendor-default sampled twin on both arms, and the 256-token spec-gate.
cell fs2-og-thinkon A2-og-thinkon.log \
  --prompts "$SH/thinkon-prompts.tsv" --goldens "$HOME/realgate/dump" \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 \
  --draft-trim "$OG" --draft-trim-n 32768 --trim-sweep 8192,16384,32768 \
  --spec-sampled --spec-gate 256 --trim-ab 5x256

# B: thinkoff, ship policy.
cell fs2-og-thinkoff B2-og-thinkoff.log \
  --prompts "$SH/thinkoff-prompts.tsv" --goldens "$HOME/realgate/dump" \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 \
  --draft-trim "$OG" --draft-trim-n 32768 \
  --spec-sampled --spec-gate 256 --trim-ab 5x256

# C: efflow, ship policy — a third real served shape, and a chain neither §2's calibration
#    nor the width choice was fitted on.
cell fs2-og-efflow C2-og-efflow.log \
  --prompts "$SH/efflow-prompts.tsv" --goldens "$HOME/realgate/dump" \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 \
  --draft-trim "$OG" --draft-trim-n 32768 \
  --spec-sampled --spec-gate 256 --trim-ab 5x256
