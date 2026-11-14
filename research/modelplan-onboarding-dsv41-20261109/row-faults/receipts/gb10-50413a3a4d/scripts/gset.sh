#!/usr/bin/env bash
# GB10, one task: TAG's tree (src-TAG.tar) built for 121a (dsv41-gate and memra-server) and its dsv41 gate set
# run at once, MEMRA_FULL_PREC=1: the kernel gate on the random fixtures, then `dsv41-gate rowfault` on the
# checkpoint's first 2 trunk layers in four arms: the resident trunk and the one-card residency (--experts-host:
# the routed experts in pinned host memory behind the device cache), each at one rank and at two ranks on the
# one card (--devices 0,0).
# usage: gset.sh TAG   Done-file out/gset-TAG/g.txt, marker GSET-DONE.
set -u
TAG=$1
R=$HOME/.cache/dsv41-fu1113
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" RUSTUP_TOOLCHAIN=1.97.1
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/gset-$TAG; mkdir -p $O; : > $O/g.txt
fin() { echo "GSET-DONE $(date +%T)" >> $O/g.txt; exit $1; }
rm -rf $R/src && mkdir -p $R/src $R/bins/$TAG && cd $R/src && tar -xf $R/src-$TAG.tar || { echo "untar failed" >> $O/g.txt; fin 1; }
find crates/memra-engine/cu crates/memra-engine/src crates/memra-engine/build.rs crates/memra-server/src -type f -exec touch {} +
echo "build $TAG start $(date +%T)" >> $O/g.txt
nice -n 19 timeout 60m cargo build --release --locked -j 16 -p memra-engine --bin dsv41-gate -p memra-server --bin memra-server > $O/build.log 2>&1
rc=$?
echo "build rc $rc $(date +%T) compiling-lines $(grep -c 'Compiling' $O/build.log)" >> $O/g.txt
if [ $rc -ne 0 ]; then grep -E "^error" -A12 $O/build.log | head -60 >> $O/g.txt; fin 1; fi
cp -a $R/target/release/dsv41-gate $R/target/release/memra-server $R/bins/$TAG/
sha256sum $R/bins/$TAG/dsv41-gate $R/bins/$TAG/memra-server >> $O/g.txt
B=$R/bins/$TAG
export MEMRA_FULL_PREC=1
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/g.txt 2>&1
timeout 20m $B/dsv41-gate kernels $D/fixtures-gpu/rand --out $O/kernels.json > $O/kernels.log 2>&1
echo "kernels rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/kernels.log | tail -1) $(grep -ac '^FAIL' $O/kernels.log) FAIL lines" >> $O/g.txt
grep -a "^FAIL" $O/kernels.log | head -20 >> $O/g.txt
# GB10's memory is the host's: drop the checkpoint's cached pages before each load
drop_cache() {
  python3 - "$C" <<'PY'
import os, sys, glob
for f in glob.glob(os.path.join(sys.argv[1], "*.safetensors")):
    fd = os.open(f, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
PY
}
for dev in 0 0,0; do
  for arm in resident host; do
    drop_cache
    free -g | head -2 | tail -1 >> $O/g.txt
    XF=""; [ $arm = host ] && XF="--experts-host"
    name=rowfault-$arm-d${dev/,/_}
    timeout 12m $B/dsv41-gate rowfault $C --token-map $TM --layers 2 $XF --devices $dev --out $O/$name.json \
      > $O/$name.log 2>&1
    rc=$?
    echo "rowfault $arm devices $dev rc $rc $(grep -o '[0-9]* checks, [0-9]* failed' $O/$name.log | tail -1)" >> $O/g.txt
    grep -a "^PASS\|^FAIL" $O/$name.log | cut -c1-400 >> $O/g.txt
  done
done
fin 0
