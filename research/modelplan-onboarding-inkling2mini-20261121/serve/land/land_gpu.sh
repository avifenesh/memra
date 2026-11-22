#!/bin/bash
# land/inkling2mini-1 on the RTX 5090 Laptop: the memra-engine GPU test binaries the dev desk cannot
# run, and the batching gate of both served classes on the landing's ink2-gpu (frozen in land-frozen).
L=/home/avifenesh/.cache/inkling2mini
B=${LAND_BIN:-$L/land-frozen2}
T=$L/oracle/tapes
M=/home/avifenesh/models/inkling2-mini
O=${LAND_GPU_OUT:-/home/avifenesh/.cache/inkling2mini/wt-serve/research/modelplan-onboarding-inkling2mini-20261121/serve/land2}
mkdir -p $O; cd $O
cat $B/COMMIT $B/SHA256SUMS
run() { tag=$1; shift; echo "MARK $tag $(date +%T.%N)"; "$@" > $tag.log 2>&1; echo "RC $tag=$?"; tail -n 2 $tag.log; }
for t in $B/tests/*; do run test-$(basename $t | sed 's/-[0-9a-f]*$//') $t --test-threads 1; done
for cls in shipped nvfp4-rig; do
  run batch-$cls $B/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out batch-$cls.json
  run batch-long-$cls $B/ink2-gpu $M --class $cls --tape $T/p2_window.exact_f32 --batch-check 4,16,256,1000,60 --out batch-long-$cls.json
done
run batch-red-shipped $B/ink2-gpu $M --class shipped --red batch-rows-rotated --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out batch-red-shipped.json
echo LAND_GPU_DONE
