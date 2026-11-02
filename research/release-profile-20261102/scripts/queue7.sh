#!/usr/bin/env bash
# Workspace bins for arm B, then the tokenizer A/B under gpulock, then cpu_native_check both arms.
set -u
cd ~/.cache/memra-relprofile
python3 measure_build.py B wsbins 1 --workspace --bins > /dev/null 2>&1
echo "wsbins B done"
gpulock run --ttl 60m --wait 90m --for "memra release-profile A/B tokenizer bench (9B NVFP4)" -- bash queue5.sh > queue5.out 2>&1
for a in A B; do
  MEMRA_CPU_EXPERT_LIB=$HOME/.cache/memra-relprofile/libmemra-cpu-experts.so CUDA_VISIBLE_DEVICES="" \
    target-$a/release/cpu_native_check > logs/cpu_native_check-$a.log 2>&1
  echo "cpu_native_check $a rc=$?"
done
echo queue7 done
