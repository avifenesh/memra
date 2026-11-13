#!/bin/bash
# Laptop: the GB10 chain on UNIT (default cb33) for the given stages: copy their tarballs, build them in one lease,
# then their gate leases (s1, s4: all; s3: kernels, tp, decode; s2: kernels), one task a lease. usage: chain.sh "1 3"
set -u
S=/home/avifenesh/.cache/dsv41-memra/cutlass-decode/fix/spark
F=/home/avifenesh/.cache/dsv41-memra/cutlass-decode/fix
export UNIT=${UNIT:-cb33}
H=spark-$UNIT
L="bash $S/lease.sh"
ST=$1  # stage numbers in gate order, e.g. "1 3"
ssh $H 'mkdir -p ~/.cache/dsv41-cfix'
for s in $ST; do scp -q $F/src-s$s.tar $H:.cache/dsv41-cfix/src-s$s.tar || { echo "copy failed; CHAIN-DONE"; exit 1; }; done
TAGS=$(for s in $ST; do printf 's%s ' $s; done)
BN=$(echo $ST | tr -d ' ')
$L "build $TAGS(121a)" out/build-$BN/runs.txt BUILD-DONE 60 build.sh "'$TAGS'" $BN
ssh $H "cat ~/.cache/dsv41-cfix/out/build-$BN/runs.txt" | grep "engine build rc"
for s in $ST; do
  ssh $H "grep -q 's$s engine build rc 0' ~/.cache/dsv41-cfix/out/build-$BN/runs.txt" || { echo "s$s BUILD RED"; continue; }
  case $s in
    1|4|5)
      $L "s$s gates A (kernels, layer default + dense form 4)" out/gates-s$s/a.txt GATES-A-DONE 30 ga.sh s$s "'d 4'"
      $L "s$s gates B1 (tp 2 ranks x3 transports, dense form 4)" out/gates-s$s/b1.txt GATES-B1-DONE 25 gb.sh s$s 1 "'0,0'" "'peer-pull host-canonical default'" 4
      $L "s$s gates B2 (tp 4 ranks peer-pull, host-canonical, dense form 4)" out/gates-s$s/b2.txt GATES-B2-DONE 25 gb.sh s$s 2 "'0,0,0,0'" "'peer-pull host-canonical'" 4
      $L "s$s gates B3 (tp 4 ranks default, dense form 4)" out/gates-s$s/b3.txt GATES-B3-DONE 20 gb.sh s$s 3 "'0,0,0,0'" "'default'" 4
      $L "s$s gates C (decode, graph door, dense form 4)" out/gates-s$s/c.txt GATES-C-DONE 25 gc.sh s$s 4
      $L "s$s order-gate tapes chat" out/tapes-s$s/chat-runs.txt TAPES-chat-DONE 22 tapes.sh s$s chat
      $L "s$s order-gate tapes long" out/tapes-s$s/long-runs.txt TAPES-long-DONE 22 tapes.sh s$s long
      ;;
    3)
      $L "s3 gates A (kernels)" out/gates-s3/a.txt GATES-A-DONE 20 ga.sh s3 none
      $L "s3 gates B1 (tp 2 ranks x3 transports)" out/gates-s3/b1.txt GATES-B1-DONE 25 gb.sh s3 1 "'0,0'" "'peer-pull host-canonical default'"
      $L "s3 gates B2 (tp 4 ranks x3 transports)" out/gates-s3/b2.txt GATES-B2-DONE 30 gb.sh s3 2 "'0,0,0,0'" "'peer-pull host-canonical default'"
      $L "s3 gates C (decode, graph door)" out/gates-s3/c.txt GATES-C-DONE 25 gc.sh s3
      ;;
    2)
      $L "s2 gates A (kernels)" out/gates-s2/a.txt GATES-A-DONE 20 ga.sh s2 none
      ;;
  esac
done
echo CHAIN-DONE
