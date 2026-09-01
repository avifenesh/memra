#!/bin/bash
# mtp10 Run E - bounded admission sweep: p-min draft-confidence guard (MEMRA_SPEC_PMIN
# semantics incl. zero-draft rounds) and adaptive K (accepted+1). End-to-end tok/s is
# the verdict metric (DRAFT-REGIME law 3); every arm interleaved 5x256 vs plain.
set -e
set -o pipefail
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp10/adm

for p in 0.3 0.5 0.7; do
  echo "=== E1: thinkon pmin=$p ==="
  target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/adm --label adm-thinkon-pmin$p \
    --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkon-prompts.tsv --mtp-dev1 \
    --spec-k 5 --spec-pmin $p --spec-ab 5x256 --spec-gate 64 --max-new 8
done

echo "=== E3: thinkon adaptive K (k_lo=1) ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/adm --label adm-thinkon-adapt1 \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkon-prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-adapt 1 --spec-ab 5x256 --spec-gate 64 --max-new 8

echo "=== E DONE ==="
