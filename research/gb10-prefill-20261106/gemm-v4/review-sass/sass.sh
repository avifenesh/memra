#!/usr/bin/env bash
# Compile one copy of mmq_nvfp4_w4a8.cu to cubins for sm_121a and sm_120a with build.rs's flags and
# dump per-kernel SASS. usage: sass.sh NAME   (reads NAME.cu next to this script)
set -eu
cd "$(dirname "$0")"
N=$1
for a in 121a 120a; do
  /usr/local/cuda/bin/nvcc -gencode "arch=compute_$a,code=sm_$a" -O3 -std=c++17 --expt-relaxed-constexpr \
    -cubin -o "$N-$a.cubin" "$N.cu"
  /usr/local/cuda/bin/cuobjdump -sass "$N-$a.cubin" > "$N-$a.sass"
  /usr/local/cuda/bin/cuobjdump -res-usage "$N-$a.cubin" > "$N-$a.res" 2>&1 || true
done
echo "SASS-DONE $N"
