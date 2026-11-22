#!/bin/bash
# which precision group makes batched rows differ from solo (shipped, p1 short check), on BIN
L=/home/avifenesh/.cache/inkling2mini
T=$L/oracle/tapes
M=/home/avifenesh/models/inkling2-mini
mkdir -p $OUT; cd $OUT
cat $BIN/COMMIT $BIN/SHA256SUMS
i=0
for p in default conv=f64 router=f64 conv=f64,router=f64 attn=f64g controller=f32,norm=f32,conv=f64,router=f64,attn=f64g controller=f64 norm=f64; do
  i=$((i+1))
  if [ $p = default ]; then a=(); else a=(--prec $p); fi
  echo "== $i $p"
  $BIN/ink2-gpu $M --class shipped "${a[@]}" --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out b$i.json > b$i.log 2>&1; echo "RC=$?"; tail -n 1 b$i.log
done
echo BISECT_DONE
