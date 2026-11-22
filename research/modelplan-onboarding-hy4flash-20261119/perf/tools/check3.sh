#!/usr/bin/env bash
# bitwise check of a binary against the PR 8 reference tapes: chat forced, long-split free, long free
# usage: check3.sh TAG BIN
set -u
tag=$1; bin=$2
cd ${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/perf
L=${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/logs/perf-check3-$tag.log
: > $L
BIN=$bin ./run-gpu.sh $tag-chatf chat --force-selections ${HY4F_ROOT:?set HY4F_ROOT to the lane run root}/pr7/oracle-set/chat; ./cmp.sh $tag-chatf chat1f >> $L 2>&1
BIN=$bin ./run-gpu.sh $tag-lsplit long-split; ./cmp.sh $tag-lsplit long-split1 >> $L 2>&1
BIN=$bin ./run-gpu.sh $tag-long long; ./cmp.sh $tag-long long1 >> $L 2>&1
echo "CHECK3_DONE $(grep -c BITWISE_EQUAL $L)/3 equal" >> $L
