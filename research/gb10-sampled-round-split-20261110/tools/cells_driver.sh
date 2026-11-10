#!/usr/bin/env bash
# Runs on the driving machine: the cells named in CELLS ("MODE:NAME ..."), one device lease per
# cell (one boot of prof_cell.sh on the GB10 host, then the fetch of its receipts). The cells of
# this lane ran through this driver with the lease tool, the unit and the host of the machine at
# hand; they are taken from the environment here.
# usage: HOST=<ssh host of the GB10 box> UNIT=<its lease unit> L=<local scratch dir> \
#        CELLS="nsys:nsys1 plain:plain1 phase:phase1 setup:setup1 burst128:burst128 nsys:nsys2 burst512:burst512" cells_driver.sh
set -u
HOST=${HOST:?}; UNIT=${UNIT:?}; L=${L:?}
T=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$L/logs" "$L/run"
for c in $CELLS; do
  MODE=${c%%:*}; NAME=${c##*:}
  # a landing's cells go first: wait while any holder or waiter of the unit is one
  waitfor --until "! sparklock status --unit $UNIT | grep -qi landing" --every 30s --timeout 40m --quiet >> "$L/logs/driver.log" 2>&1 \
    || { echo "$(date -Is) $NAME: a landing held the unit for 40m, stopping" >> "$L/logs/driver.log"; exit 1; }
  echo "$(date -Is) $NAME: queue" >> "$L/logs/driver.log"
  sparklock run --unit "$UNIT" --ttl 20m --wait 40m --for "sampled-round split, cell $NAME (one boot, six requests)" -- bash -c "
    scp -q '$T/prof_cell.sh' '$T/req_driver.py' '$HOST:.cache/memra-1186/box/' || exit 1
    ssh '$HOST' 'bash ~/.cache/memra-1186/box/prof_cell.sh $MODE $NAME' 2>&1 | tee '$L/logs/cell-$NAME.out'
    mkdir -p '$L/run/$NAME'
    rsync -a --exclude prof.nsys-rep --max-size=3g '$HOST:.cache/memra-1186/run/$NAME/' '$L/run/$NAME/'
  " > "$L/logs/lease-$NAME.log" 2>&1
  echo "$(date -Is) $NAME: rc=$?" >> "$L/logs/driver.log"
done
echo "$(date -Is) DRIVER-DONE" >> "$L/logs/driver.log"
