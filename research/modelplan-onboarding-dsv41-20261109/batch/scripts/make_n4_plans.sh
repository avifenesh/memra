#!/bin/bash
# The four-rank plans from the lane's existing job lines: n4a (device-push arm) and n4b (peer-pull arm).
P=$LANE/spec/plans
line() { grep -h "^$1 $2 " $P/$3.plan | head -n 1; }
{
  line batch-rows chat8 gate9b
  line batch-rows nine gate9b
  line batch-time distinct8 gate9b
  line batch-rows window gate9b
  line batch-rows mixed gate9b
  line batch-time distinct16 wide9
} > $P/n4a.plan
{
  line batch-rows chat8 gate9b
  line batch-rows wide16 wide9
} > $P/n4b.plan
wc -l $P/n4a.plan $P/n4b.plan
cut -c1-60 $P/n4a.plan $P/n4b.plan
