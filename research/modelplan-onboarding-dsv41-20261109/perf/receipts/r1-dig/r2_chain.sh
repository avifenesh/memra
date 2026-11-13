#!/bin/bash
# Laptop: r2's GB10 proof on b23d, three leases one after the other (fresh acquire each, so others interleave):
#   a: build 121a + kernel gate + layer tensors vs k3's set; chat: the 13 chat tapes, byte check vs k3's;
#   b: tp at 2 and 4 ranks + decode on three captures. Stops after a failed part.
set -u
S=/home/avifenesh/.cache/dsv41-memra/r1-dig/spark
bash $S/r1dig_lease.sh b23d "r2: build 121a + kernel gate + layer tensors vs k3's set" r2_gates.sh "a" \
  out/gates-r2/gates-a.txt GATES-R2-DONE 30 || exit 1
ssh spark-b23d 'grep -q "111 compared, 0 differ" ~/.cache/dsv41-speed/out/gates-r2/gates-a.txt' || { echo "R2-CHAIN-STOP a"; exit 1; }
bash $S/r1dig_lease.sh b23d "r2: chat tapes, byte check vs k3's" r2_tapes.sh "chat" out/r2-tapes/runs-chat.txt \
  R2-TAPES-DONE 30 || exit 1
bash $S/r1dig_lease.sh b23d "r2: tp 2/4 ranks + decode gates" r2_gates.sh "b" out/gates-r2/gates-b.txt GATES-R2-DONE 30
echo "R2-CHAIN-DONE"
