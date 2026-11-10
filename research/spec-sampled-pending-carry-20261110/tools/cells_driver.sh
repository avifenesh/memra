#!/usr/bin/env bash
# Runs on the driving machine: the GB10 runs of this lane, one gate run or one boot per device
# lease. The lease tool, the unit and the host are the machine's own and come from the
# environment; a lease is taken only when the unit is free and nothing queued on it is a
# landing.
# usage: HOST=<ssh host of the GB10 box> UNIT=<its lease unit> H=<local scratch dir> W=<this tree> \
#        cells_driver.sh PHASE...
#   A  the base binary: the invariance gate with the device penalty arm, the end-of-request
#      cell (two boots)
#   B  build the candidate; the invariance gate on it at the default and with the device
#      penalty arm; the burst-boundary gate on the base binary, then on the candidate against
#      the base's rows; the end-of-request cell (two boots)
#   C  the timed cell: 12 boots, base cand cand base x3
# The base binary's invariance gate at the default ran first, by hand, in the same way.
set -u
HOST=${HOST:?}; UNIT=${UNIT:?}; H=${H:?}; W=${W:?}
L=$H/logs; mkdir -p "$L"
T=$W/research/spec-sampled-pending-carry-20261110/tools
DEV="--penalised device --env MEMRA_SERVE_DEVPENALTY=1"
lease() { # NAME TTL WHAT CMD...
  local name=$1 ttl=$2 what=$3; shift 3
  waitfor --until "sparklock status --unit $UNIT | grep -q '^$UNIT: free' && ! sparklock status --unit $UNIT | grep -qi landing" \
    --every 20s --timeout 240m --quiet >> "$L/driver.log" 2>&1 || return 1
  sparklock run --unit "$UNIT" --ttl "$ttl" --wait 3m --for "sampled pending carry: $what" -- "$@" > "$L/lease-$name.log" 2>&1
}
remote() { # SUB 'remote command': copy the tools, run, fetch the run directory
  scp -q "$T"/gb10-*.sh "$W/tools/seeded-invariance-gate.py" "$W/tools/spec-burst-boundary-gate.py" \
    "$T/ab_cell.py" "$T/close_cell.py" "$HOST:.cache/memra-carry/box/" || return 1
  ssh "$HOST" "$2"; local rc=$?
  mkdir -p "$H/run/gb10/$1"
  rsync -a --exclude texts "$HOST:.cache/memra-carry/run/$1/" "$H/run/gb10/$1/"
  return $rc
}
export -f remote; export HOST H W T
gate() { # NAME ARM TTL WHAT TOOL ARGS...
  local name=$1 arm=$2 ttl=$3 what=$4 tool=$5; shift 5
  lease "$name" "$ttl" "$what" bash -c "remote $name 'bash ~/.cache/memra-carry/box/gb10-gate.sh $name $arm $tool $*'"
  echo "$name exit $?" >> "$L/driver.rc"
}
boot() { # NAME PROG SUB ARM BOOT TTL WHAT ENV...
  local name=$1 prog=$2 sub=$3 arm=$4 n=$5 ttl=$6 what=$7; shift 7
  lease "$name" "$ttl" "$what" bash -c "remote $sub 'bash ~/.cache/memra-carry/box/gb10-boot.sh $prog $sub $arm $n $*'"
  echo "$name exit $?" >> "$L/driver.rc"
}
for phase in "$@"; do
  case $phase in
    A)
      gate invdev-base base 50m "invariance gate with the device penalty arm, base" seeded-invariance-gate.py $DEV
      boot close-base close_cell.py close base 1 25m "end-of-request cell, base" MODE=close
      boot demote-base close_cell.py close base 1 25m "end-of-request cell with a forced demotion, base" MODE=demote
      ;;
    B)
      lease build 25m "build the candidate" bash -c "ssh $HOST 'mkdir -p ~/.cache/memra-carry/tree ~/.cache/memra-carry/box ~/.cache/memra-carry/run' && rsync -a --delete --exclude .git --exclude target --exclude research $W/ $HOST:.cache/memra-carry/tree/ && scp -q $T/gb10-*.sh $HOST:.cache/memra-carry/box/ && ssh $HOST 'bash ~/.cache/memra-carry/box/gb10-build.sh cand'" || exit 1
      gate inv-cand cand 50m "invariance gate, candidate" seeded-invariance-gate.py --penalised host
      gate invdev-cand cand 50m "invariance gate with the device penalty arm, candidate" seeded-invariance-gate.py $DEV
      gate boundary-base base 55m "burst-boundary gate, base" spec-burst-boundary-gate.py --carry no $DEV
      gate boundary-cand cand 55m "burst-boundary gate, candidate" spec-burst-boundary-gate.py $DEV --compare '$HOME/.cache/memra-carry/run/boundary-base/runs.jsonl'
      boot close-cand close_cell.py close cand 1 25m "end-of-request cell, candidate" MODE=close
      boot demote-cand close_cell.py close cand 1 25m "end-of-request cell with a forced demotion, candidate" MODE=demote
      ;;
    C)
      n=0
      for arm in base cand cand base base cand cand base base cand cand base; do
        n=$((n + 1))
        boot "perf-$n" ab_cell.py perf "$arm" "$n" 15m "timed cell boot $n of 12 ($arm)" MODE=perf
      done
      ;;
  esac
done
