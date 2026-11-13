#!/bin/bash
export MEMRA_FULL_PREC=1
gate() {  # NAME ENV...
  local name=$1; shift
  env "$@" /home/access/dsv41/perf-bins/kpc/dsv41-gate tp /home/access/dsv41/DeepSeek-V4.1-Flash-1017 /home/access/dsv41/captures-v2/needle --layers 1,2,14,20,21,24 --devices 0,1 --decode 2     --token-map /home/access/dsv41/oracle/engram_constants.token_map.i32 --out /home/access/dsv41/receipts-perf/pairgates-kpc-23/tp-$name.json > /home/access/dsv41/receipts-perf/pairgates-kpc-23/tp-$name.log 2>&1
  echo "tp $name rc $? $(grep -o '[0-9]* checks, [0-9]* failed' /home/access/dsv41/receipts-perf/pairgates-kpc-23/tp-$name.log | tail -1) $(grep -ac '^FAIL' /home/access/dsv41/receipts-perf/pairgates-kpc-23/tp-$name.log) FAIL lines" >> /home/access/dsv41/receipts-perf/pairgates-kpc-23/gates.txt
  grep -a "stress/\|graph/replayed" /home/access/dsv41/receipts-perf/pairgates-kpc-23/tp-$name.log | cut -c1-220 >> /home/access/dsv41/receipts-perf/pairgates-kpc-23/gates.txt
}
: > /home/access/dsv41/receipts-perf/pairgates-kpc-23/gates.txt
gate default MEMRA_UNUSED=1
gate peer-pull MEMRA_TP_TRANSPORT=peer-pull
gate host-canonical MEMRA_TP_TRANSPORT=host-canonical
gate graph MEMRA_DSV41_DECODE_GRAPH=1
