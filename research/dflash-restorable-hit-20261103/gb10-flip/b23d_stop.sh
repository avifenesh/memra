#!/usr/bin/env bash
# Runs ON spark-b23d. The landing agent needs b23d right after arm 3, so b23d_arms_local.sh stops after
# arm 3: once arm 3's server is gone, hold port 8090 so the chain's next serve refuses ("port 8090
# busy") and exits before it boots anything, then free the port. The c1 tail runs later under a new lease.
set -u
D=$HOME/memra-run
L=$D/arms-local/chain.log
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) [stop] $*" | tee -a "$L"; }
for i in $(seq 1 1440); do grep -q 'done dflash-flip-default' "$L" && break; sleep 5; done
grep -q 'done dflash-flip-default' "$L" || { log "arm 3 never finished"; exit 2; }
pid=$(cat $D/serve-bs-gguf-dflash.pid)
for i in $(seq 1 1500); do [ -e /proc/$pid ] || break; sleep 0.2; done
python3 -m http.server 8090 --bind 127.0.0.1 > $D/port-hold.log 2>&1 &
holder=$!
log "arm 3 server $pid gone; holding 8090 (pid $holder)"
for i in $(seq 1 120); do grep -q 'refusing' "$L" && break; sleep 5; done
kill $holder
grep -q 'refusing' "$L" && log "B23D_STOPPED after arm 3" || log "chain did not refuse"
