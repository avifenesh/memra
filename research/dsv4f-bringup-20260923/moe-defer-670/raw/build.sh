#!/bin/bash
# build.sh <tree> <target-dir> <tag>
set -uo pipefail
tree=$1; tgt=$2; tag=$3
. /root/.cargo/env
export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd "$tree"
log=/root/build-$tag.log
{ echo "BUILD $tag head=$(git rev-parse HEAD) utc=$(date -u +%FT%TZ)"
  CARGO_TARGET_DIR=$tgt cargo build --release -j 48 -p memra-server --bins
  echo "SERVER_EXIT=$?"
  CARGO_TARGET_DIR=$tgt cargo build --release -j 48 -p memra-engine --bins
  echo "ENGINE_EXIT=$?"
  echo "DONE utc=$(date -u +%FT%TZ)"; } > "$log" 2>&1
