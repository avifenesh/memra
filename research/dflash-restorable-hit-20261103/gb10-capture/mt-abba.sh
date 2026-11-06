#!/usr/bin/env bash
# Runs ON spark-b23d under the caller's lease: the multi-turn persistence A/B of the cold-drafter
# capture fix, order main/fix/fix/main, fresh boot each, gguf-dflash arm with the restore doors on
# by device default and MEMRA_PREFIX_STABLE_BOUNDARY=0, client on the unit (multiturn_client.py).
set -u
LEASE=${LEASE:?lease}
D=$HOME/memra-run
O=$D/coldcap/mt
mkdir -p "$O"
L=$D/coldcap/chain.log
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) [mt] $*" | tee -a "$L"; }
export SPARKLOCK_LEASE=$LEASE
DEF="MEMRA_DSPARK_PREFIX_RESTORE= MEMRA_DSPARK_PARTIAL_RESTORE= MEMRA_DSPARK_BOUNDARY_CAPTURE= MEMRA_PREFIX_STABLE_BOUNDARY=0"
n=0
for arm in main fix fix main; do
  n=$((n+1)); tag=$n-$arm
  bin=$D/bin-main-3bbdf42ff9; [ $arm = fix ] && bin=$D/bin-fix-c28fb5ee0eaf
  if ss -ltn | grep -q ':8090 '; then log "port 8090 busy before $tag; refusing"; exit 4; fi
  (ARM=gguf-dflash BIN=$bin LEASE=$LEASE EXTRA="$DEF" nohup bash $D/serve-batchspec.sh) > $D/serve-mt-$tag.out 2>&1 < /dev/null &
  ok=0
  for i in $(seq 1 80); do curl -sf http://127.0.0.1:8090/v1/models >/dev/null && { ok=1; break; }; sleep 15; done
  [ $ok = 1 ] || { log "boot FAILED $tag"; exit 3; }
  log "start $tag ($(sha256sum $bin/memra-server | cut -c1-12))"
  timeout 3000 python3 $D/multiturn_client.py http://127.0.0.1:8090 q38 4 "$O/turns.jsonl" "$tag" > "$O/$tag.client.log" 2>&1 \
    || log "$tag client rc=$?"
  pid=$(cat $D/serve-bs-gguf-dflash.pid)
  kill "$pid"
  for i in $(seq 1 90); do [ -e /proc/$pid ] || break; sleep 2; done
  cp $D/serve-bs-gguf-dflash.log "$O/$tag.server.log"
  sleep 3
  cp $D/guard-bs-gguf-dflash.log "$O/$tag.guard.log"
  f="$O/$tag.server.log"
  log "$tag veto=$(grep -a -c cache-preferred $f) conv=$(grep -a -c 'cold-drafter restore' $f) restore=$(grep -a -c 'DSPARK restore' $f) boundary=$(grep -a -c 'insert (dspark-boundary)' $f)"
done
log MT_DONE
