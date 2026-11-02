#!/usr/bin/env bash
# CPU suites (build timed, then run) and the qualified gate-binary set, both arms.
set -u
cd ~/.cache/memra-relprofile
export CUDA_VISIBLE_DEVICES=""
CPU="-p memra-reference -p memra-sampling -p memra-tokenizer -p memra-gguf -p memra-validate --lib"
GATE="-p memra-engine --bin kernel-check --bin run-gen --bin run-spec --bin argmax-margin-probe -p memra-tokenizer --bin tok-parity"
for arm in A B; do
  python3 measure_build.py $arm testbuild-cpu 1 $CPU > /dev/null 2>&1
  python3 measure_build.py $arm testrun-cpu 1 $CPU > /dev/null 2>&1
  python3 measure_build.py $arm testbuild-engine 1 -p memra-engine --lib > /dev/null 2>&1
  python3 measure_build.py $arm testrun-engine 1 -p memra-engine --lib > /dev/null 2>&1
  python3 measure_build.py $arm gatebins 1 $GATE > /dev/null 2>&1
  python3 measure_build.py $arm cpu-native-check-build 1 -p memra-engine --bin cpu_native_check > /dev/null 2>&1
done
echo queue3 done
