#!/usr/bin/env bash
# Host-side serving A/B on the 5090, interleaved ABBAAB, holding the GPU lease and the
# rig build lock so no build heats the CPU during a round.
set -u
cd ~/.cache/memra-relprofile
cat /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf > /dev/null
i=0
for arm in A B B A A B; do
  i=$((i+1))
  flock ~/.cache/memra-rig-build.lock python3 host_bench.py $arm $i > logs/host-$arm-$i.out 2>&1 || echo "round $i arm $arm failed rc=$?"
done
echo queue2 done
