#!/bin/bash
# mtp10 Run E2 - admission combination arms: adaptive K x p-min guard on thinkon, and
# the guard tax on the WINNING shape (thinkoff must hold its 1.50x).
set -e
set -o pipefail
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git log -1 --format="HEAD %H"
mkdir -p ~/realgate/mtp10/adm

echo "=== E2: thinkon adapt1 + pmin0.3 ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/adm --label adm-thinkon-a1p03 \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkon-prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 --spec-ab 5x256 --spec-gate 64 --max-new 8

echo "=== E2: thinkon adapt1 + pmin0.5 ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/adm --label adm-thinkon-a1p05 \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkon-prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.5 --spec-ab 5x256 --spec-gate 64 --max-new 8

echo "=== E2: thinkoff adapt1 (guard tax on the winning shape) ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/adm --label adm-thinkoff-adapt1 \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkoff-prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-adapt 1 --spec-ab 5x256 --spec-gate 64 --max-new 8

echo "=== E2: thinkoff adapt1 + pmin0.5 ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/adm --label adm-thinkoff-a1p05 \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkoff-prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.5 --spec-ab 5x256 --spec-gate 64 --max-new 8

echo "=== E2 DONE ==="
