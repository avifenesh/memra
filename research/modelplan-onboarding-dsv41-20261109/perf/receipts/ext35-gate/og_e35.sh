#!/bin/bash
# Dev desk: the by-tape order gate (order_gate.py, the same file og_cutlass.sh ran) of one build family against the
# current route over the decode lane's 18 vendor tapes, the 35 ext35 tapes, and both together (53).
# Inputs under ~/.cache/dsv41-og/e35: all/{tapes,ref,floor}/ID.pt (symlinks: the 18 from ../tapes and ../x/dec, the 35
# from e35/{tapes,ref,floor}), b/BUILD/ID/ (the engine's teacher-forced files of every case, from wrx
# receipts-perf/ext35-BUILD/{ext35,chat,long,long257} or the earlier ordertapes dirs).
# usage: bash og_e35.sh CUR NEW   -> e35/og/NEW-vs-CUR-{18,35,53}.{md,json,err}
set -u
cd ~/.cache/dsv41-og || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CUR=$1; NEW=$2
O=e35/og; mkdir -p $O
P="nice -n 10 venv/bin/python order_gate.py --tapes e35/all/tapes --ref e35/all/ref --floor e35/all/floor --cur e35/b/$CUR --new e35/b/$NEW"
A18=capital,gsm,code,copy,hebrew,gsm_think,story,hebrew_para,code_pal,sys_json,multiturn,needle,needle1200,decode32k_he_summary,decode64k_en_continue,needle4800,needle8400,needle16800
X35=$(python3 -c 'import json; print(",".join(c["id"] for c in json.load(open("e35/ext35-oracle.json"))))')
for s in 18 35 53; do
  case $s in 18) L=$A18 ;; 35) L=$X35 ;; 53) L=$A18,$X35 ;; esac
  $P --cases $L --json $O/$NEW-vs-$CUR-$s.json > $O/$NEW-vs-$CUR-$s.md 2> $O/$NEW-vs-$CUR-$s.err
  echo "$NEW vs $CUR $s rc $? $(date +%T)"
done
echo OG-E35-DONE $NEW
