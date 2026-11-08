#!/usr/bin/env bash
# Runs ON cb33: stop this lane's fup-gates.sh run (its process group) before the 05:50 lease end,
# then run the pp3855 logits trio on the follow-up binary.
set -u
D=$HOME/.cache/memra-f8f4v3
pg=$(ps -eo pgid=,args= | awk '/[f]up-gates\.sh/ {print $1; exit}')
if [ -n "$pg" ]; then
  echo "stopping process group $pg"
  kill -TERM -- "-$pg"
fi
waitfor --until 'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"' --every 2s --timeout 40s 2>&1 | tail -n 1
ps -eo pgid=,pid=,args= | awk -v g="$pg" '$1 == g' | head -n 5
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
bash "$D/logits3.sh" "$D/bins/v3-292b21bdf8" "$D/runs/fup-292b21bdf8/gates"
