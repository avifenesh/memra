#!/usr/bin/env bash
set -euo pipefail
export PATH=$HOME/.cargo/bin:$PATH
cd /root/memra-release-v01340
nvcc -o /root/release-v01340-work/hold research/mtp-head-nonresident-capture-20260908/hold.cu
holder=
runner=
cleanup() {
  if [ -n "$runner" ] && kill -0 "$runner" 2>/dev/null; then kill "$runner"; wait "$runner" 2>/dev/null || true; fi
  if [ -n "$holder" ]; then kill "$holder" 2>/dev/null || true; wait "$holder" 2>/dev/null || true; fi
}
trap cleanup EXIT HUP INT TERM
/root/release-v01340-work/hold 10240 > /root/release-v01340-work/holder.log 2>&1 &
holder=$!
echo "$holder" >> /root/release-v01340-work/owned-pids
for i in $(seq 1 30); do grep -q '^HELD' /root/release-v01340-work/holder.log && break; sleep 1; done
cat /root/release-v01340-work/holder.log
grep -q '^HELD' /root/release-v01340-work/holder.log
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
MEMRA_GPU_LOCK= /root/target/release/run-spec /data/qual/models/orn-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf &
runner=$!
echo "$runner" >> /root/release-v01340-work/owned-pids
wait "$runner"
runner=
