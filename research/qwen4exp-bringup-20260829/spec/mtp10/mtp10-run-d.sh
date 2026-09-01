#!/bin/bash
# mtp10 Run F2 (phase-windowed nsys: the verify-vs-plain kernel identity) then Run D
# (the scaled own-gen rank corpus; resumable via the corpus ledger).
set -e
set -o pipefail
export PATH=$HOME/.cargo/bin:$PATH
cd ~/memra
git log -1 --format="HEAD %H"
sha256sum target/release/qwen4exp_real_gate
mkdir -p ~/realgate/mtp10/nsys ~/realgate/mtp10/corpus

echo "=== F2a: PLAIN decode window (t=1 kernel sums) ==="
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -o ~/realgate/mtp10/nsys/win-plain --force-overwrite true \
  target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/nsys --label win-plain \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkon-prompts.tsv --mtp-dev1 \
  --decode-timing 40 --profiler-window --max-new 8
nsys stats --report cuda_gpu_kern_sum --format csv -o ~/realgate/mtp10/nsys/win-plain ~/realgate/mtp10/nsys/win-plain.nsys-rep || true

echo "=== F2b: SPEC rounds window (verify + draft kernel sums, thinkon, K=5 fixed, no admission) ==="
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -o ~/realgate/mtp10/nsys/win-spec --force-overwrite true \
  target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/nsys --label win-spec \
  --goldens ~/realgate/dump --prompts ~/realgate/mtp9/shapes/thinkon-prompts.tsv --mtp-dev1 \
  --spec-k 5 --spec-ab 1x128 --profiler-window --max-new 8
nsys stats --report cuda_gpu_kern_sum --format csv -o ~/realgate/mtp10/nsys/win-spec ~/realgate/mtp10/nsys/win-spec.nsys-rep || true
echo "=== F2 DONE ==="

echo "=== Run D: scaled own-gen corpus ==="
target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/mtp10/corpus --label owngen-mtp10 \
  --mtp-dev1 --spec-k 5 \
  --owngen ~/realgate/mtp10/corpus-prompts-big.tsv --owngen-out ~/realgate/mtp10/ranks-owngen-big.txt \
  --owngen-corpus-out ~/realgate/mtp10/corpus-ids-big.tsv \
  --owngen-greedy 256 --owngen-sampled 512 --owngen-seeds 2
echo "=== D DONE ==="
