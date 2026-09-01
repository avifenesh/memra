#!/bin/bash
# mtp9 Run C — the vgraph seam exactness gate: verify rows bit-identical and spec chains
# byte-identical with the segment graphs FORCED ON. $1 = trim N (0 = full-vocab head).
set -e
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
N=${1:-0}
export MEMRA_Q4E_SEAMS=vgraph
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp9/vgraph-gate
TRIM=""
[ "$N" != "0" ] && TRIM="--draft-trim $HOME/realgate/mtp9/ranks-owngen.txt --draft-trim-n $N"
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/vgraph-gate --label vgraph-on-trim$N \
  --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv $TRIM \
  --spec-k 5 --verify-bit-gate 24 --spec-gate 64 --max-new 8
