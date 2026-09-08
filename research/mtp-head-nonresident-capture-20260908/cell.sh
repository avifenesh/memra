#!/bin/bash
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/root/target MEMRA_CUDA_ARCH=120a MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
set -u
cd /root/memra || exit 1
mib=$1
label=$2
holder=
cleanup() { if [ -n "$holder" ]; then kill "$holder" 2>/dev/null; wait "$holder" 2>/dev/null; fi; }
trap cleanup EXIT
{
  git -c safe.directory=/root/memra rev-parse --short HEAD
  git -c safe.directory=/root/memra status --short | head
  sha256sum /root/target/release/run-spec
  if [ "$mib" -gt 0 ]; then
    /root/mtpcap/hold "$mib" > "/root/mtpcap/holder-$label.log" 2>&1 &
    holder=$!
    echo "$holder" >> /root/mtpcap/owned-pids
    for i in $(seq 1 30); do grep -q HELD "/root/mtpcap/holder-$label.log" && break; sleep 1; done
    cat "/root/mtpcap/holder-$label.log"
    grep -q '^HELD' "/root/mtpcap/holder-$label.log" || exit 2
  fi
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
  MEMRA_GPU_LOCK= /root/target/release/run-spec /data/qual/models/orn-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf &
  runner=$!
  echo "$runner" >> /root/mtpcap/owned-pids
  wait "$runner"
  rc=$?
  echo "RUN_EXIT=$rc"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
  exit "$rc"
} > "/root/mtpcap/$label.log" 2>&1
