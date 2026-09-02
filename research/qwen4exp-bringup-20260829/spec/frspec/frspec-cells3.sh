#!/usr/bin/env bash
# FR-Spec trim battery, part 3 on the shipped program (memra#61): the bench-shape pair and
# the class control, kept in a SEPARATE file from cells2 so neither script is ever edited
# while bash is reading it (see the note in frspec-cells2.sh).
#
# D: raw fixed K=5 (the shape mtp10 measured at 0.8824 with N=11,854) — bench-only: an
#    11-token friendly prompt, and any degenerate continuation is a greedy-loop artifact,
#    never a finding. It is here because it is the banked comparison point.
# E: the SAME shape and width with the PURE-CORPUS ranks instead of the own-gen-headed
#    blend — the class control. Same prompt, same width, same policy, one variable.
set -u
export CUDA_VISIBLE_DEVICES=0
export MEMRA_Q4E_SEAMS=idxsel,kvq,idxq
BIN=$HOME/realgate/bin/qwen4exp_real_gate.frspec2
CKPT=$HOME/data/q48fn-yarn1m
OUT=$HOME/realgate/frspec
OG=$OUT/q4e-ranks-ogblend-32768.txt
SXC=$OUT/q4e-ranks-sxc32768.txt
LK=/tmp/q48fn-measure.lock
Q=$OUT/QUEUE.log
mkdir -p "$OUT"
say(){ echo "[$(date -u +%FT%TZ)] frspec3: $*" >> "$Q"; }

cell(){ # cell <label> <logfile> <args...>
  local label=$1 log=$2; shift 2
  say "$label acquiring -x"
  flock -x "$LK" -c "$HOME/realgate/bin/q4e-load-lock.sh $OUT/$log $BIN $CKPT $OUT --label $label $*"
  say "$label rc=$?"
}

cell fs2-og-raw D2-og-raw.log \
  --prompts "$HOME/realgate/dump/prompts.tsv" \
  --spec-k 5 \
  --draft-trim "$OG" --draft-trim-n 32768 \
  --spec-gate 256 --trim-ab 5x256

cell fs2-sxc-raw E2-sxc-raw.log \
  --prompts "$HOME/realgate/dump/prompts.tsv" \
  --spec-k 5 \
  --draft-trim "$SXC" --draft-trim-n 32768 \
  --spec-gate 256 --trim-ab 5x256
