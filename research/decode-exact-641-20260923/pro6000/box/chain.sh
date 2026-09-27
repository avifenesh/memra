#!/usr/bin/env bash
set -uo pipefail
E=/root/e641
cd /root/wt-e || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> $E/chain.log; }
log "chain start tree=$(git rev-parse HEAD)"
deadline=$((SECONDS + 1800))
while ! flock -n /tmp/memra-gpu.lock true; do
  [ $SECONDS -ge $deadline ] && { log "lock still held after 1800 s; not run"; exit 3; }
  sleep 30
done
log "gpu cells start"
python3 tools/tier-battery.py --rig pro-single --timeout 3600 --out $E/collector --external-lock --execute \
  bash $E/gpu.sh @COLLECTOR_LOCK_FD@ > $E/collector.log 2>&1
log "gpu cells exit=$?"
log "chain done"
