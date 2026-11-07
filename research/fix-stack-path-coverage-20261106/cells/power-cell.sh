#!/bin/bash
# usage: power-cell.sh GPU WATTS TAG LOGDIR -- CMD...
# Owner ruling 2026-11-05: a memra Max-Q cell may raise its card to WATTS (300 W), must note thermal throttling and
# must restore the limit after use. This wrapper sets the limit, runs CMD in its own process group, samples the card
# every 10 s (temperature, fan, power, clock-event reasons) into LOGDIR/thermal-TAG.csv, stops the cell on the Max-Q
# thermal rule (>= 95 C, or >= 92 C with fan >= 95 % on 3 consecutive samples), restores the limit it found and
# writes one summary line to LOGDIR/power-TAG.txt. Exit status is CMD's (143 after a thermal stop).
# Neighbour rule (asked by the box's training session for the GPU 2 gap cells): if POWER_DIR/neighbours-gpuG exists
# ("1,3 90"), and any listed card reaches the threshold, the card returns to its previous limit for the rest of the cell.
# REASSERT=1: the box re-applies its own caps every 10 minutes (gpu-cap.timer); for a cell registered at WATTS the
# wrapper sets WATTS again on the first sample that reads another limit, and the summary counts the re-sets.
set -u
G=$1; PW=$2; TAG=$3; D=$4; shift 4
[ "${1:-}" = -- ] && shift
NVSMI=${NVSMI:-nvidia-smi}
SUDO=${SUDO-sudo -n}
P=${POWER_DIR:-<maxq-kit>/power}
ORIGF=$P/orig-gpu$G
CSV=$D/thermal-$TAG.csv
SUM=$D/power-$TAG.txt
NB=""; NBMAX=999
[ -f "$P/neighbours-gpu$G" ] && read -r NB NBMAX < "$P/neighbours-gpu$G"
q() { $NVSMI -i "$G" --query-gpu="$1" --format=csv,noheader,nounits | tr -d ' ' | cut -d. -f1; }

# A leftover orig file means an earlier cell died before restoring: its value is the limit to return to.
if [ -f "$ORIGF" ]; then orig=$(cat "$ORIGF"); else orig=$(q power.limit); echo "$orig" > "$ORIGF"; fi
maxl=$(q power.max_limit)
want=$PW
[ "$want" -gt "$maxl" ] && want=$maxl
$SUDO $NVSMI -i "$G" -pl "$want" > "$D/power-set-$TAG.log" 2>&1
got=$(q power.limit)

restore() {
  $SUDO $NVSMI -i "$G" -pl "$orig" >> "$D/power-set-$TAG.log" 2>&1
  local now
  now=$(q power.limit)
  if [ "$now" = "$orig" ]; then rm -f "$ORIGF"; echo "restored ${orig} W"; else echo "RESTORE FAILED: reads ${now} W, wanted ${orig} W"; fi
}

echo "timestamp,temperature_c,fan_pct,power_draw_w,power_limit_w,clocks_sm_mhz,sw_power_cap,sw_thermal_slowdown,hw_thermal_slowdown,hw_power_brake,neighbour_max_c" > "$CSV"
setsid "$@" &
cp=$!
trap 'kill -TERM -- -$cp 2>/dev/null' TERM INT

n=0; hot=0; tmax=0; fmax=0; c_pc=0; c_swt=0; c_hwt=0; c_pb=0; stop=""; nbmax=0; nbdrop=""; c_lw=0; loff=""; c_ra=0
num() { case "$1" in ''|*[!0-9]*) echo 0 ;; *) echo "$1" ;; esac; }
while kill -0 "$cp" 2>/dev/null; do
  r=$($NVSMI -i "$G" --query-gpu=timestamp,temperature.gpu,fan.speed,power.draw,power.limit,clocks.sm,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown --format=csv,noheader,nounits 2>&1)
  nbt=""
  if [ -n "$NB" ]; then
    nbt=$($NVSMI -i "$NB" --query-gpu=index,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -t, -k2 -n | tail -n 1)
    nbi=${nbt%%,*}; nbc=$(num "${nbt#*,}")
    [ "$nbc" -gt "$nbmax" ] && nbmax=$nbc
    if [ -z "$nbdrop" ] && [ "$nbc" -ge "$NBMAX" ]; then
      $SUDO $NVSMI -i "$G" -pl "$orig" >> "$D/power-set-$TAG.log" 2>&1
      nbdrop="$(date +%T) GPU $nbi at $nbc C -> ${orig} W"
      echo "$(date +%F_%T) neighbour rule: $nbdrop for the rest of the cell" | tee -a "$D/power-set-$TAG.log"
    fi
    nbt=$nbc
  fi
  echo "$r, $nbt" >> "$CSV"
  IFS=',' read -r ts t f _ lim _ pc swt hwt pb <<< "$r"
  t=$(num "${t// /}"); f=$(num "${f// /}")
  # The limit each sample reads: the box re-applies its own caps on a timer, so the asked limit may not hold.
  lim=${lim// /}; lim=${lim%%.*}
  if [ "$lim" = "$want" ]; then c_lw=$((c_lw + 1)); elif [ -z "$loff" ]; then loff="${ts# } -> ${lim} W"; fi
  # REASSERT=1 (a cell registered at WATTS, e.g. a timed half): set the asked limit again when a sample reads another
  # one, unless the neighbour rule or the thermal stop has taken the card back. The off sample still counts as off.
  if [ "${REASSERT:-0}" = 1 ] && [ "$lim" != "$want" ] && [ -z "$nbdrop" ] && [ -z "$stop" ]; then
    $SUDO $NVSMI -i "$G" -pl "$want" >> "$D/power-set-$TAG.log" 2>&1
    c_ra=$((c_ra + 1))
    echo "$(date +%F_%T) re-asserted ${want} W (sample read ${lim} W)" >> "$D/power-set-$TAG.log"
  fi
  n=$((n + 1))
  [ "$t" -gt "$tmax" ] && tmax=$t
  [ "$f" -gt "$fmax" ] && fmax=$f
  [ "${pc// /}" = Active ] && c_pc=$((c_pc + 1))
  [ "${swt// /}" = Active ] && c_swt=$((c_swt + 1))
  [ "${hwt// /}" = Active ] && c_hwt=$((c_hwt + 1))
  [ "${pb// /}" = Active ] && c_pb=$((c_pb + 1))
  if [ -z "$stop" ]; then
    if [ "$t" -ge 92 ] && [ "$f" -ge 95 ]; then hot=$((hot + 1)); else hot=0; fi
    if [ "$t" -ge 95 ]; then stop="temperature ${t} C >= 95 (fan ${f} %)"
    elif [ "$hot" -ge 3 ]; then stop="temperature ${t} C with fan ${f} % on 3 consecutive 10 s samples"; fi
    if [ -n "$stop" ]; then
      echo "$(date +%F_%T) THERMAL-STOP GPU $G: $stop" | tee "$D/THERMAL-STOP-$TAG.txt"
      kill -TERM -- -"$cp" 2>/dev/null
      kdl=$(( $(date +%s) + 60 ))
    fi
  elif [ "$(date +%s)" -ge "$kdl" ]; then
    kill -KILL -- -"$cp" 2>/dev/null
  fi
  sleep "${SAMPLE_S:-10}" & wait $!
done
wait "$cp"; rc=$?
[ -n "$stop" ] && rc=143
res=$(restore)
throttled=no
ra_txt=""; [ "${REASSERT:-0}" = 1 ] && ra_txt=", re-asserted ${c_ra} times"
[ $((c_swt + c_hwt + c_pb)) -gt 0 ] && throttled=THERMAL-THROTTLED
line="power GPU $G: limit ${got} W (asked ${want}, was ${orig}, max ${maxl}); ${n} samples, ${c_lw} at ${want} W${loff:+, first sample off it ${loff}}${ra_txt}; max ${tmax} C, max fan ${fmax} %; sw_thermal ${c_swt}, hw_thermal ${c_hwt}, hw_power_brake ${c_pb}, sw_power_cap ${c_pc}; throttle: ${throttled}; thermal stop: ${stop:-none}; neighbours ${NB:-none} max ${nbmax} C, drop: ${nbdrop:-none}; ${res}; rc=${rc}"
echo "$line" | tee "$SUM"
exit $rc
