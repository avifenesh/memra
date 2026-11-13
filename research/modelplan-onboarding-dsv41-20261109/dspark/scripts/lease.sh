#!/bin/bash
# One single-card GPU lease for the DSpark drafting lane on the run box. QUIET and PAUSE checks, the box lock and the card
# lease through memra-gpu-run, cores 24-31,56-63, receipts under leases/spec-<name>-<time>, the card's power limit
# recorded before the run (never changed here).
# usage: lease.sh NAME TIMEOUT_S CARD CMD...     CARD: 1 or 3
# Heat rule (coordinator, until ~05:10 IST): a GPU1 step starts only when GPU2's fan is at or under 90%.
set -uo pipefail
NAME=$1; TO=$2; CARD=$3; shift 3
D=${LANE:?LANE names the working directory on the run box (checkpoint, oracle, dec ids, spec/, target-spec/)}
Q=$QUIET_FILE
if [ -e $Q ]; then echo "QUIET is set: not starting $NAME"; exit 3; fi
if [ -e $D/spec/PAUSE ]; then echo "spec/PAUSE is set: not starting $NAME"; exit 3; fi
case "$CARD" in 1|3) ;; *) echo "CARD: 1 or 3"; exit 2 ;; esac
if [ "$CARD" = 1 ]; then
  fan=$(nvidia-smi -i 2 --query-gpu=fan.speed --format=csv,noheader,nounits | tr -d ' ')
  if [ "${fan:-100}" -gt 90 ]; then echo "GPU2 fan ${fan}%: not starting $NAME on GPU1"; exit 5; fi
fi
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$CARD" '$1==i{print $2}')
R=$D/leases/spec-$NAME-$(date +%Y%m%dT%H%M%S)
mkdir -p "$R"
nvidia-smi --query-gpu=index,uuid,power.limit,temperature.gpu,fan.speed,clocks.gr --format=csv > "$R/cards.csv"
echo "$(date +%H:%M:%S) $NAME: lease on GPU$CARD ($UUID), receipt $R"
$D/bin/memra-gpu-run --gpus "$UUID" --receipt "$R" --box-lock $BOX_LOCKS/gpu$CARD.lock \
  --timeout "$TO" --wait 1800 -- taskset -c 24-31,56-63 "$@"
rc=$?
echo "$(date +%H:%M:%S) $NAME: lease ended rc $rc"
exit $rc
