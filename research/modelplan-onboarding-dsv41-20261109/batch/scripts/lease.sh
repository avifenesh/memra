#!/bin/bash
# One single-card GPU lease for the DSpark drafting lane on the run box. QUIET and PAUSE checks, the box lock and the card
# lease through memra-gpu-run, cores SPEC_CPUS (default 28-31,60-63: the lane's half while another agent runs), receipts under leases/spec-<name>-<time>, the card's power limit
# recorded before the run (never changed here).
# usage: lease.sh NAME TIMEOUT_S CARD CMD...     CARD: 1 or 3
# Heat rule: a GPU1 step starts only when GPU2's fan is at or under 90% (it waits up to 20 min for that).
set -uo pipefail
NAME=$1; TO=$2; CARD=$3; shift 3
D=$LANE
Q=$QUIET_FILE
if [ -e $Q ]; then echo "QUIET is set: not starting $NAME"; exit 3; fi
if [ -e $D/spec/PAUSE ]; then echo "spec/PAUSE is set: not starting $NAME"; exit 3; fi
case "$CARD" in 0|1|3) ;; *) echo "CARD: 0, 1 or 3"; exit 2 ;; esac
if [ "$CARD" = 1 ]; then
  # heat: no GPU1 step starts while GPU2's fan is above 90%; the step waits for it (bounded) instead of being dropped
  fan_ok() {
    fan=$(nvidia-smi -i 2 --query-gpu=fan.speed --format=csv,noheader,nounits | tr -d ' ')
    [ "${fan:-100}" -le 90 ]
  }
  tries=0
  until fan_ok; do
    [ $tries -eq 0 ] && echo "$(date +%H:%M:%S) $NAME: GPU2 fan ${fan}%, waiting up to 20 min to start on GPU1"
    if [ $tries -ge 40 ]; then echo "GPU2 fan still above 90%: not starting $NAME on GPU1"; exit 5; fi
    tries=$((tries + 1))
    /bin/sleep 30
    if [ -e $Q ]; then echo "QUIET is set: not starting $NAME"; exit 3; fi
  done
fi
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$CARD" '$1==i{print $2}')
R=$D/leases/spec-$NAME-$(date +%Y%m%dT%H%M%S)
mkdir -p "$R"
nvidia-smi --query-gpu=index,uuid,power.limit,temperature.gpu,fan.speed,clocks.gr --format=csv > "$R/cards.csv"
echo "$(date +%H:%M:%S) $NAME: lease on GPU$CARD ($UUID), receipt $R"
$D/bin/memra-gpu-run --gpus "$UUID" --receipt "$R" --box-lock $LOCK_DIR/gpu$CARD.lock \
  --timeout "$TO" --wait 1800 -- taskset -c ${SPEC_CPUS:-28-31,60-63} "$@"
rc=$?
echo "$(date +%H:%M:%S) $NAME: lease ended rc $rc"
exit $rc
