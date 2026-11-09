#!/usr/bin/env bash
# Compare the cell record's cached binaries of two commit pairs. usage: bincmp.sh BINS TARGET OLD NEW [OLD NEW ...]
set -u
BINS=$1; TGT=$2; shift 2
while [ $# -ge 2 ]; do
  o=$1; n=$2; shift 2
  for b in argmax-margin-probe kernel-check memra-server run-gen run-spec tok-parity; do
    if cmp -s "$BINS/$o-$TGT/$b" "$BINS/$n-$TGT/$b"; then r=identical; else r="differ($(cmp -l "$BINS/$o-$TGT/$b" "$BINS/$n-$TGT/$b" | wc -l) bytes)"; fi
    echo "${o:0:10} vs ${n:0:10} $b: $r"
  done
done
