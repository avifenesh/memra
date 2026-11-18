#!/usr/bin/env bash
# Build host: compile cu/dsv41_gpu.cu (build.rs flags) of src-base and src-new for the non-120 build arches
# (89, 90a, 100a); dsv41_cutlass.cu is replaced by its stub there. Marker arch-out/DONE.
set -u
B=$HOME/.cache/bs-small
O=$B/arch-out; mkdir -p "$O"; rm -f "$O/DONE"; : > "$O/rc.txt"
for tag in base new; do
  for arch in 89 90a 100a; do
    (cd "$B/src-$tag/crates/memra-engine" && taskset -c 14-31 nice -n 10 /usr/local/cuda-13.1/bin/nvcc \
      -gencode "arch=compute_${arch},code=sm_${arch}" -O3 -std=c++17 --expt-relaxed-constexpr -fmad=false \
      -c cu/dsv41_gpu.cu -o "$O/dsv41_gpu-$arch-$tag.o") > "$O/$tag-$arch.log" 2>&1
    echo "$tag $arch rc=$?" >> "$O/rc.txt"
  done
done
touch "$O/DONE"
