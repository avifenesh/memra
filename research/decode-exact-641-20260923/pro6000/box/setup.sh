#!/usr/bin/env bash
# lane E part 2, box setup (CPU only): own clone, base (main d544c6b82) and fix (lane tip) builds.
set -uo pipefail
E=/root/e641
mkdir -p $E/bins/base $E/bins/fix
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $E/setup.log; }
BASE_SHA=d544c6b82
LANE=lane/decode-exact-641-20260923
if [ ! -d /root/wt-e/.git ]; then
    URL=$(git -C /root/wt-b remote get-url origin)
    log "clone from origin (objects borrowed from the lane-B checkout, then dissociated)"
    git clone -q --reference /root/wt-b --dissociate "$URL" /root/wt-e >> $E/setup.log 2>&1 || { log "clone failed"; exit 1; }
fi
cd /root/wt-e || exit 1
git fetch -q origin main "$LANE" >> $E/setup.log 2>&1 || { log "fetch failed"; exit 1; }
FIX_SHA=$(git rev-parse "origin/$LANE")
log "base $(git rev-parse $BASE_SHA) fix $FIX_SHA"

git checkout -q --detach $BASE_SHA || exit 1
log "build base start"
( cargo build --release -p memra-engine --bin prime-batch-gate --offline \
  || cargo build --release -p memra-engine --bin prime-batch-gate ) > $E/build-base.log 2>&1
rc=$?; log "build base rc=$rc"; [ $rc = 0 ] || exit 1
cp target/release/prime-batch-gate $E/bins/base/
git rev-parse HEAD > $E/bins/base/source.txt

git checkout -q --detach "$FIX_SHA" || exit 1
log "build fix start"
( cargo build --release -p memra-server -p memra-engine --bin memra-server --bin prime-batch-gate --bin concat-prime-probe --offline \
  || cargo build --release -p memra-server -p memra-engine --bin memra-server --bin prime-batch-gate --bin concat-prime-probe ) > $E/build-fix.log 2>&1
rc=$?; log "build fix rc=$rc"; [ $rc = 0 ] || exit 1
cp target/release/prime-batch-gate target/release/concat-prime-probe target/release/memra-server $E/bins/fix/
git rev-parse HEAD > $E/bins/fix/source.txt
sha256sum $E/bins/base/* $E/bins/fix/* | tee $E/bins.sha256
log "setup done"
