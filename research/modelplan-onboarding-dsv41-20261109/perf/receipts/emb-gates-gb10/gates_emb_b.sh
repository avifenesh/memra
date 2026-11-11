#!/usr/bin/env bash
# GB10 lease B (<= 30 min): the paths the change runs through, emb against kpc on the same inputs.
#  1. dsv41-run --cases (capital, code_pal, needle 3,761 tokens; teacher-forced) over the first 4 layers (Engram
#     layer 1 inside) on one rank and on two ranks of the card: every output file of emb byte-equal to kpc's
#     (Runner::forward's prefill and one-token calls now take their rows from the table on the card).
#  2. dsv41-tpbench, 4 layers resident on one rank (load_resident: the table, the pool reserve; Runner::new: the
#     pinned staging), prefills of 3,761 and 18,343 tokens, 2 rounds, --nvtx parsed: argmax equal, TTFT a reading.
#  3. the tp gate at 2 and 4 ranks on the card, the default transport.
set -u
R=$HOME/.cache/dsv41-speed
D=$R/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
O=$R/out/gates-emb
mkdir -p $O
: > $O/gates-b.txt
export MEMRA_FULL_PREC=1
I=$D/dec/ids
printf '%s\n' "capital $I/capital.prompt.i32 $I/capital.gen.i32" "code_pal $I/code_pal.prompt.i32 $I/code_pal.gen.i32" \
  "needle $I/needle.prompt.i32 $I/needle.gen.i32" > $O/cases.txt
for dev in 0 0,0; do
  for arm in kpc emb; do
    rm -rf $O/run-$arm-$dev; mkdir -p $O/run-$arm-$dev
    t=$(date +%s)
    timeout 6m $R/bins/$arm/dsv41-run $C --token-map $TM --cases $O/cases.txt --layers 4 --devices $dev \
      --out $O/run-$arm-$dev > $O/run-$arm-$dev.log 2>&1
    echo "run $arm devices $dev rc $? $(( $(date +%s) - t )) s $(grep -c . $O/run-$arm-$dev.log) log lines" >> $O/gates-b.txt
  done
  n=0; bad=0; : > $O/diff-$dev.txt
  for f in $(cd $O/run-kpc-$dev && find . -type f | sort); do
    n=$((n+1)); cmp -s $O/run-kpc-$dev/$f $O/run-emb-$dev/$f || { bad=$((bad+1)); echo "$f" >> $O/diff-$dev.txt; }
  done
  echo "devices $dev: emb against kpc, $n files compared, $bad differ ($(head -c 300 $O/diff-$dev.txt | tr '\n' ' '))" >> $O/gates-b.txt
done
n=0; bad=0
for f in $(cd $O/run-emb-0 && find . -type f -name '*.f32' -o -type f -name '*.i32' | sort); do
  n=$((n+1)); cmp -s $O/run-emb-0/$f $O/run-emb-0,0/$f || bad=$((bad+1))
done
echo "emb one rank against two ranks: $n value files compared, $bad differ" >> $O/gates-b.txt
P=$I/needle.prompt.i32,$I/needle1200.prompt.i32
for arm in kpc emb; do
  fl=""; [ $arm = emb ] && fl=--nvtx
  rm -rf $O/tpb-$arm; mkdir -p $O/tpb-$arm
  timeout 5m $R/bins/$arm/dsv41-tpbench $C --token-map $TM --out $O/tpb-$arm --devices 0 --layers 4 --resident 4 \
    --prefill $P --rounds 2 $fl > $O/tpb-$arm.log 2>&1
  echo "tpbench $arm rc $? $(grep -c '"kind":"prefill"' $O/tpb-$arm/cells.jsonl 2>/dev/null) cells" >> $O/gates-b.txt
  grep -a "dsv41-resident\|capacity at" $O/tpb-$arm.log | head -n 3 | cut -c1-220 >> $O/gates-b.txt
  python3 - $O/tpb-$arm/cells.jsonl >> $O/gates-b.txt <<'PY'
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("kind") == "prefill":
        print(f"  {r['cell']} round {r['round']}: argmax {r['argmax']} ttft {r['ttft_s'] * 1e3:.1f} ms forward {r['forward_s'] * 1e3:.1f} ms")
PY
done
for dev in 0,0 0,0,0,0; do
  timeout 7m $R/bins/emb/dsv41-gate tp $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --devices $dev --decode 2 \
    --token-map $TM --out $O/tp-$dev.json > $O/tp-$dev.log 2>&1
  echo "tp $dev default rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/tp-$dev.log | tail -1)" >> $O/gates-b.txt
done
echo "GATES-EMB-B-DONE $(date +%T)" >> $O/gates-b.txt
