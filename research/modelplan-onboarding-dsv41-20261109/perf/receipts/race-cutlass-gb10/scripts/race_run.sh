#!/usr/bin/env bash
# Race hunt, GB10 run lease: the tp gate, 2 ranks on one card, peer-pull, layers 2 and 24, through the plan
# file $PLAN (lines: "TAG BINDIR REPS DECODE ENV..."), in order, until done or the deadline (DEADLINE_MIN minutes
# after start; no run starts after it). Out dir out/$OUT; done-file out/$OUT/runs.txt, marker RACE-RUN-DONE.
set -u
R=$HOME/.cache/dsv41-cutlass
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
OUT=$1; PLAN=$2; DEADLINE_MIN=${3:-26}
export MEMRA_FULL_PREC=1 MEMRA_TP_TRANSPORT=peer-pull
O=$R/out/$OUT; mkdir -p $O; : > $O/runs.txt; rm -f $O/STOP
END=$(( $(date +%s) + DEADLINE_MIN * 60 ))
while read -r tag bin reps dec envs; do
  [ -z "$tag" ] && continue
  case $tag in \#*) continue;; esac
  if [ $(date +%s) -ge $END ]; then echo "DEADLINE: stopped before $tag $(date +%T)" >> $O/runs.txt; break; fi
  if [ -e $O/STOP ]; then echo "STOP (yield): stopped before $tag $(date +%T)" >> $O/runs.txt; break; fi
  t0=$(date +%s)
  timeout 8m env V41DBG_REPS=$reps $envs $R/bins/$bin/dsv41-gate tp $C $D/captures-v2/needle --layers 2,24 \
    --devices 0,0 --decode $dec --token-map $TM --out $O/$tag.json > $O/$tag.log 2>&1 < /dev/null
  rc=$?
  nf=$(grep -ac '^FAIL' $O/$tag.log)
  echo "$tag rc $rc $(grep -ao '[0-9]* checks, [0-9]* failed' $O/$tag.log | tail -1) fails $nf secs $(( $(date +%s) - t0 )) $(date +%T)" >> $O/runs.txt
  grep -a '^FAIL' $O/$tag.log | head -4 >> $O/runs.txt
done < $R/$PLAN
echo "RACE-RUN-DONE $(date +%T)" >> $O/runs.txt
