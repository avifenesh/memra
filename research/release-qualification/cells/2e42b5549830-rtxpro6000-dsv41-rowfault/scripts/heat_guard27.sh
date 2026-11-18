#!/usr/bin/env bash
# the PRO 6000 box, landing 27: the Max-Q heat rule for one leased card. Samples temperature and fan every 10 s while the lease
# wrapper PID lives. Stops the lease (SIGTERM to the wrapper, which stops its whole group) only when the card reads
# >= 92 C AND fan >= 95% in three samples in a row. A hot card with fan headroom is normal on these cards.
# usage: heat_guard27.sh GPU_INDEX WRAPPER_PID LOG
set -u
G=$1; P=$2; LOG=$3
n=0
nvidia-smi -i "$G" --query-gpu=temperature.gpu,fan.speed --format=csv,noheader,nounits -l 10 | while IFS=', ' read -r t f; do
  kill -0 "$P" 2> /dev/null || break
  if [ "$t" -ge 92 ] && [ "$f" -ge 95 ]; then n=$((n + 1)); else n=0; fi
  echo "$(date '+%H:%M:%S') card $G ${t} C fan ${f}% hot-run $n" >> "$LOG"
  if [ "$n" -ge 3 ]; then
    echo "$(date '+%H:%M:%S') HEAT-STOP: >= 92 C and fan >= 95% three samples in a row; SIGTERM to wrapper $P" >> "$LOG"
    kill -TERM "$P" 2> /dev/null
    break
  fi
done
