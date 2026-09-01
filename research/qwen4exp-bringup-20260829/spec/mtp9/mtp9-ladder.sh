#!/bin/bash
# mtp9 Run B — the K ladder at the trimmed config (a cheap draft is what makes bigger K
# profitable) plus the vendor-default SAMPLED spec-engagement receipt.
#   $1 = trim N   $2 = spec K for the A/B + sampled probe   $3 = ladder   $4 = seams
# Split into narrow ladders on purpose: the verify stash grows with k_cap (~1 GiB at K=8
# on ~2.6 GiB of post-load headroom) and the ladder receipt is only written after the
# LAST K, so one OOM at the top would cost every row below it.
set -e
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
N=${1:?usage: mtp9-ladder.sh <trim_n> <spec_k> <ladder> [seams]}
K=${2:?usage: mtp9-ladder.sh <trim_n> <spec_k> <ladder> [seams]}
L=${3:?usage: mtp9-ladder.sh <trim_n> <spec_k> <ladder> [seams]}
[ -n "$4" ] && export MEMRA_Q4E_SEAMS="$4"
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp9/ladder
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp9/ladder \
  --label ladder-trim$N-k$K-L$(echo $L | tr , _)${4:+-$4} \
  --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv \
  --draft-trim ~/realgate/mtp9/ranks-owngen.txt --draft-trim-n $N \
  --spec-k $K --spec-ab 5x256 --spec-ladder $L \
  --spec-sampled --decode-timing 40 --max-new 8
