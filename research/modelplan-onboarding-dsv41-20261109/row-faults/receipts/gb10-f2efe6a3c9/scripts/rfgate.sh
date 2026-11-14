#!/usr/bin/env bash
# GB10 `dsv41-gate rowfault` of TAG's binary on the checkpoint's first 2 trunk layers, MEMRA_FULL_PREC=1: the
# resident trunk, then the served one-card residency (--experts-host: the routed experts in pinned host memory
# behind the device cache). usage: rfgate.sh TAG   Done-file out/rowfault-TAG/r.txt, marker RFGATE-DONE.
set -u
TAG=$1
R=$HOME/.cache/dsv41-fu1113
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
B=$R/bins/$TAG
O=$R/out/rowfault-$TAG; mkdir -p $O; : > $O/r.txt
sha256sum $B/dsv41-gate >> $O/r.txt
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/r.txt 2>&1
for arm in resident host; do
  # GB10's memory is the host's: drop the checkpoint's cached pages before each load
  python3 - "$C" <<'PY'
import os, sys, glob
for f in glob.glob(os.path.join(sys.argv[1], "*.safetensors")):
    fd = os.open(f, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
PY
  free -g | head -2 | tail -1 >> $O/r.txt
  XF=""; [ $arm = host ] && XF="--experts-host"
  timeout 15m $B/dsv41-gate rowfault $C --token-map $TM --layers 2 $XF --out $O/rowfault-$arm.json \
    > $O/rowfault-$arm.log 2>&1
  rc=$?
  echo "rowfault $arm rc $rc $(grep -o '[0-9]* checks, [0-9]* failed' $O/rowfault-$arm.log | tail -1)" >> $O/r.txt
  grep -a "^PASS\|^FAIL" $O/rowfault-$arm.log >> $O/r.txt
done
echo "RFGATE-DONE $(date +%T)" >> $O/r.txt
