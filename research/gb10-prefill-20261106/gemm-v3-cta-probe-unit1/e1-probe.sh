#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's sparklock lease: the fused gate/up silu launch on v2 against
# v3 with the CTA-local TMA destination (C = the landed TU with that one change), in one process, at
# T 2048 and 4096, both in-process orders, twice.
# usage: e1-probe.sh OUTDIR
set -u
OUT=$1
D=$HOME/.cache/memra-f8f4v3/cta
export PATH="/usr/local/cuda/bin:$PATH" TMPDIR=$HOME/.cache/memra-f8f4v3/tmp
mkdir -p "$OUT"
(cd "$D/C" && nice -n 10 nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr \
   -o drv drv_cta.cu -lcuda > "$OUT/build-C.log" 2>&1); echo "build C rc=$?"
test -x "$D/C/drv" || { echo BUILD-FAIL; exit 1; }
(cd "$D" && sha256sum C/mmq_nvfp4_w4a8.cu C/drv_cta.cu C/drv) > "$OUT/SHA256SUMS"
n=0
for T in 2048 4096; do
  for r in 1 2; do
    for ord in 01 10; do
      n=$((n+1))
      f=$OUT/$(printf %02d $n)-T$T-C-$ord.log
      "$D/C/drv" "$T" 11 "$ord" > "$f" 2>&1
      echo "== C T=$T order=$ord"; grep -E "ms|rc=|cuda:" "$f"
    done
  done
done
echo E1-PROBE-DONE
