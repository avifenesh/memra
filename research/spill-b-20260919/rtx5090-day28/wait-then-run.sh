#!/usr/bin/env bash
# Bounded wait for the local card to free (memory.used < 2500 MiB (the card idle but for a long-lived 1.4 GB tenant) for two consecutive 30 s polls, up to 90 min), then the
# day-28 cell. Never inspects or signals another process; the cell itself takes /tmp/memra-5090.lock.
set -u
R=$HOME/projects/wt-spill-b/research/spill-b-20260919/rtx5090-day28; cell=${1:-before}
ok=0; for i in $(seq 180); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "$(date -u +%FT%TZ) memory.used=${u} MiB" >> "$R/$cell.wait.log"
  if [ "${u:-99999}" -lt 2500 ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 2 ] && break; sleep 30
done
[ $ok -ge 2 ] || { echo "$(date -u +%FT%TZ) card not free within 90 min; cell not run" | tee -a "$R/$cell.wait.log"; exit 3; }
exec bash "$HOME/projects/wt-spill-b/research/spill-b-20260919/run-day28-cell.sh" "$cell"
