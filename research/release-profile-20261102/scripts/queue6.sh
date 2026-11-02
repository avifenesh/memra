#!/usr/bin/env bash
# Whole-workspace binaries (the CI build job's full-mode command and docs/INSTALLATION.md's
# `cargo build --release`), from the state left by the cold memra-server build. Arm A runs
# while the GPU is held by another session; arm B runs after the tokenizer bench, then the
# cpu_native_check numeric comparison on both arms.
set -u
cd ~/.cache/memra-relprofile
python3 measure_build.py A wsbins 1 --workspace --bins > /dev/null 2>&1
echo "wsbins A done"
waitfor --grep 'queue5 done' queue5.out --timeout 3h --every 30s --quiet
python3 measure_build.py B wsbins 1 --workspace --bins > /dev/null 2>&1
for a in A B; do
  MEMRA_CPU_EXPERT_LIB=$HOME/.cache/memra-relprofile/libmemra-cpu-experts.so CUDA_VISIBLE_DEVICES="" \
    target-$a/release/cpu_native_check > logs/cpu_native_check-$a.log 2>&1
  echo "cpu_native_check $a rc=$?"
done
echo queue6 done
