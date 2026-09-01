#!/bin/bash
# mtp10 Run G - FR-Spec trim retry at the prior lanes corpus scale (owner lever 2):
# topN 32768 from the ~450k-token own-gen corpus. G1 = the mtp9 twin cell (raw, fixed
# K=5, out-of-class) so the corpus-scale delta is isolated; G2/G3 = the serving cells at
# the ship policy; G4 = width sweep. Waits for Run D.
set -e
set -o pipefail
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
while pgrep -f "owngen-mtp10" > /dev/null; do sleep 120; done
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
test -s ~/realgate/mtp10/ranks-owngen-big.txt || { echo "NO RANKS - D did not finish"; exit 1; }
mkdir -p ~/realgate/mtp10/trim
grep -m1 "corpus" ~/realgate/mtp10/corpus/owngen-owngen-mtp10.tsv || true

echo "=== G1: raw goldens, fixed K=5, trim 32768 (the mtp9 -16.6% twin cell) ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/trim --label trim-raw-k5 \
  --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv --mtp-dev1 \
  --spec-k 5 --draft-trim ~/realgate/mtp10/ranks-owngen-big.txt --draft-trim-n 32768 \
  --trim-ab 5x256 --spec-gate 64 --max-new 8

echo "=== G2: thinkon at ship policy, trim 32768 ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/trim --label trim-thinkon-ship \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkon-prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 \
  --draft-trim ~/realgate/mtp10/ranks-owngen-big.txt --draft-trim-n 32768 \
  --trim-ab 5x256 --spec-gate 64 --max-new 8

echo "=== G3: thinkoff at ship policy, trim 32768 ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/trim --label trim-thinkoff-ship \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkoff-prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 \
  --draft-trim ~/realgate/mtp10/ranks-owngen-big.txt --draft-trim-n 32768 \
  --trim-ab 5x256 --spec-gate 64 --max-new 8

echo "=== G4: raw width sweep 16384/32768/65536, fixed K=5 ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/trim --label trim-sweep-raw \
  --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv --mtp-dev1 \
  --spec-k 5 --draft-trim ~/realgate/mtp10/ranks-owngen-big.txt \
  --trim-sweep 16384,32768,65536 --spec-ab 1x256 --max-new 8

echo "=== G DONE ==="
