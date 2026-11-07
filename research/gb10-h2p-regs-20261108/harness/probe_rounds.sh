#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: probe3 timing + byte compare rounds.
# usage: probe_rounds.sh TAG "KERNELS" [ROUNDS] [DEPTHS]
#   KERNELS: comma list; the first is the reference (h2p). Round r > 1 alternates the order of
#   the candidates (reference first in every round). Each round runs T = 4,096 rows at DEPTHS,
#   median of 3 after a warmup per (kernel, depth).
set -u
TAG=$1; KS=$2; ROUNDS=${3:-3}; DEPTHS=${4:-4096,16384,32768,65536,131072}
D=$HOME/.cache/memra-attn3
P=$D/probe
L=$D/logs/probe-$TAG.log
REF=${KS%%,*}; REST=${KS#*,}
rev=$(echo "$REST" | tr ',' '\n' | tac | paste -sd, -)
{
  echo "# $(date -Is) $(hostname) probe_rounds TAG=$TAG ROUNDS=$ROUNDS DEPTHS=$DEPTHS"
  sha256sum "$P"/flash_attn.cu "$P"/attn3.cuh "$P"/attn3b.cuh "$P"/probe3.cu "$P"/probe3 | sed 's/^/# /'
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader | sed 's/^/# /'
  for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) = 1 ]; then order="$REF,$REST"; else order="$REF,$rev"; fi
    echo "# round $r order $order"
    "$P/probe3" 4096 "$DEPTHS" 3 "$order"
  done
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader | sed 's/^/# end /'
  echo "PROBE-DONE $TAG"
} > "$L" 2>&1
tail -n 1 "$L"
