#!/usr/bin/env bash
# Build host: build cu/dsv41_gpu.cu of src-TAG for sm_120a only (object and PTX), for quick SASS/PTX checks.
# usage: quick.sh TAG
set -u
TAG=$1
B=$HOME/.cache/bs-small
S=$B/src-$TAG/crates/memra-engine
O=$B/q-$TAG
NVCC=/usr/local/cuda-13.1/bin/nvcc
mkdir -p "$O"
rm -f "$O/DONE" "$O/FAIL"
cd "$S" || exit 1
F=(-gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 --expt-relaxed-constexpr -fmad=false)
taskset -c 14-31 nice -n 10 "$NVCC" "${F[@]}" -c cu/dsv41_gpu.cu -o "$O/dsv41_gpu-120a.o" > "$O/build.log" 2>&1
rc=$?
taskset -c 14-31 nice -n 10 "$NVCC" -arch=compute_120a -O3 -std=c++17 --expt-relaxed-constexpr -fmad=false -ptx \
  cu/dsv41_gpu.cu -o "$O/dsv41_gpu-120a.ptx" >> "$O/build.log" 2>&1
rc2=$?
echo "rc=$rc ptx=$rc2" > "$O/rc.txt"
if [ $rc -eq 0 ] && [ $rc2 -eq 0 ]; then touch "$O/DONE"; else touch "$O/FAIL"; fi
