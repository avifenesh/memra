#!/usr/bin/env bash
# Build host: compile cu/dsv41_gpu.cu and cu/dsv41_cutlass.cu with build.rs's flags for sm_120a and sm_121a,
# and leave a marker (SASS is dumped on the laptop: the build host has no cuobjdump). usage: nv_build.sh TAG  (sources in ~/.cache/bs-small/src-TAG)
set -u
TAG=$1
B=$HOME/.cache/bs-small
S=$B/src-$TAG/crates/memra-engine
O=$B/obj-$TAG
NVCC=/usr/local/cuda-13.1/bin/nvcc
mkdir -p "$O"
rm -f "$O/DONE" "$O/FAIL"
cd "$S" || exit 1
rc_all=0
for arch in 120a 121a; do
  gc="arch=compute_${arch},code=sm_${arch}"
  for unit in dsv41_gpu dsv41_cutlass; do
    extra=()
    if [ "$unit" = dsv41_gpu ]; then extra=(-fmad=false); fi
    if [ "$unit" = dsv41_cutlass ]; then
      extra=(-DNDEBUG -I "$S/third_party/cutlass/include" -I "$S/third_party/cutlass/tools/util/include")
    fi
    t0=$(date +%s)
    taskset -c 14-31 nice -n 10 "$NVCC" -gencode "$gc" -O3 -std=c++17 --expt-relaxed-constexpr "${extra[@]}" \
      -c "cu/$unit.cu" -o "$O/$unit-$arch.o" > "$O/$unit-$arch.log" 2>&1
    rc=$?
    echo "$unit $arch rc=$rc $(( $(date +%s) - t0 ))s" | tee -a "$O/build.txt"
    if [ $rc -ne 0 ]; then rc_all=1; continue; fi
  done
done
(cd "$O" && sha256sum ./*.o > objs.sha256)
if [ $rc_all -eq 0 ]; then touch "$O/DONE"; else touch "$O/FAIL"; fi
