#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's sparklock lease: the v3 TMA destination probe. drvA links the
# landed TU (.shared::cluster destination), drvB the follow-up TU (.shared::cta destination, v2 moved
# onto the shared stage/epilogue functions; v2 SASS identical). Each binary times v2 and v3 per launch
# in one process; the two binaries run interleaved A B A B in both in-process orders.
# usage: cta-probe.sh OUTDIR
set -u
OUT=$1
D=$HOME/.cache/memra-f8f4v3/cta
export PATH="/usr/local/cuda/bin:$PATH" TMPDIR=$HOME/.cache/memra-f8f4v3/tmp
mkdir -p "$OUT"
for v in A B; do
  (cd "$D/$v" && nice -n 10 nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr \
     -o drv drv_cta.cu -lcuda > "$OUT/build-$v.log" 2>&1; echo "build $v rc=$?") &
done
wait
for v in A B; do test -x "$D/$v/drv" || { echo "BUILD-FAIL $v"; exit 1; }; done
(cd "$D" && sha256sum A/mmq_nvfp4_w4a8.cu B/mmq_nvfp4_w4a8.cu A/drv B/drv) > "$OUT/SHA256SUMS"
n=0
for T in 2048 4096; do
  for r in 1 2; do
    for ord in 01 10; do
      for v in A B; do
        n=$((n+1))
        f=$OUT/$(printf %02d $n)-T$T-$v-$ord.log
        "$D/$v/drv" "$T" 11 "$ord" > "$f" 2>&1
        echo "== $v T=$T order=$ord"; grep -E "ms|rc=|cuda:" "$f"
      done
    done
  done
done
echo CTA-PROBE-DONE
