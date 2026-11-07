#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: byte-compare every candidate with h2p on the real prime
# inputs dumped by dump_run.sh (one dir per (depth, attention layer)), reps 3.
# usage: real_probe.sh TAG KERNELS
set -u
TAG=$1; KS=$2
D=$HOME/.cache/memra-attn3
L=$D/logs/real-$TAG.log
{
  echo "# $(date -Is) $(hostname) real-data probe KS=$KS"
  for dir in "$D"/dumps/t*-n*; do
    echo "# dump $(basename "$dir") meta $(cat "$dir/meta.txt") q $(sha256sum < "$dir/q.bin" | cut -c1-16) k $(sha256sum < "$dir/k.bin" | cut -c1-16) v $(sha256sum < "$dir/v.bin" | cut -c1-16)"
    PROBE_LOAD=$dir PROBE_CUBIN=$D/probe/x134.cubin "$D/probe/probe3" - - 3 "$KS" | grep -E "^T "
  done
  echo "REAL-DONE $TAG"
} > "$L" 2>&1
grep -c "mismatch 0 nonfinite 0" "$L"; grep -E "mismatch [1-9]|nonfinite [1-9]" "$L" | head; tail -n 1 "$L"
