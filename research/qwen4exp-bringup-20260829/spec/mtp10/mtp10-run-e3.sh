#!/bin/bash
# mtp10 Run E3 - the SHIP battery at the candidate admission policy (adaptive K k_lo=1
# + p-min 0.3): every shape interleaved 5x256 vs plain + spec-gate byte identity +
# vendor-default sampled probe (serving law). The bar: NO shape regresses vs plain.
# Then Run D resumes (the corpus ledger keeps every banked generation).
set -e
set -o pipefail
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
while pgrep -f mtp10-run-e2.sh > /dev/null; do sleep 30; done
git pull --ff-only origin qwen4exp-bringup-20260829 2>&1 | tail -1
cargo build --release -p memra-engine --bin qwen4exp_real_gate 2>&1 | tail -1
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp10/ship

echo "=== E3: raw goldens (must hold its 1.79x class) ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/ship --label ship-raw \
  --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 --spec-ab 5x256 --spec-gate 64 --spec-sampled --max-new 8

for shape in thinkon thinkoff efflow; do
  echo "=== E3: $shape at adapt1+pmin0.3 ==="
  target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/ship --label ship-$shape \
    --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/$shape-prompts.tsv --mtp-dev1 \
    --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 --spec-ab 5x256 --spec-gate 64 --spec-sampled --max-new 8
done

echo "=== E3: long agentic prompt (724 tokens) at adapt1+pmin0.3 ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/ship --label ship-long \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp10/long-prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-adapt 1 --spec-pmin 0.3 --spec-ab 5x256 --spec-gate 128 --max-new 8

echo "=== E3 DONE ==="
echo "=== resuming Run D (corpus) ==="
~/mtp10-run-d.sh
