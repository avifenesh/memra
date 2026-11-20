#!/usr/bin/env bash
# b23d: bitwise check of a binary against the PR 8 reference tapes: chat forced, long-split free, long free
# usage: check3.sh TAG BIN
set -u
tag=$1; bin=$2
cd $HOME/hy4flash/perf
L=$HOME/hy4flash/logs/perf-check3-$tag.log
: > $L
BIN=$bin ./run-gpu.sh $tag-chatf chat --force-selections $HOME/hy4flash/pr7/oracle-set/chat; ./cmp.sh $tag-chatf chat1f >> $L 2>&1
BIN=$bin ./run-gpu.sh $tag-lsplit long-split; ./cmp.sh $tag-lsplit long-split1 >> $L 2>&1
BIN=$bin ./run-gpu.sh $tag-long long; ./cmp.sh $tag-long long1 >> $L 2>&1
echo "CHECK3_DONE $(grep -c BITWISE_EQUAL $L)/3 equal" >> $L
