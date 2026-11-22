#!/bin/bash
# the box, 2026-11-22: landing 30's four-card cells for memra#1337 (2df9c40b39, perf-bins l30a1337) and memra#1343
# (868417afce, perf-bins l30b1343), all the box builds (nvcc 13.4, sm_120a).
#   1. #1337 vs main 91e01a7dcc (perf-bins l30main): perf_cfix_four.sh with B=l30a1337: kernel gate (both rings), tp gates
#      on three transports (four ranks), then the TIMED A/B, order A B B A B A A B, 4 8 9 10 12 16 rows + one-row decode.
#   2. main's kernel gate on card 0 (the check count #1337's is compared with).
#   3. #1343's battery phases kgate (card 0) and tp4 (four ranks), BASE l30a1337, HEAD l30b1343.
# Starts on /mnt/Data/work/evals/dssparks.window-0123 newer than this runner's armed marker and 02:30, with all four
# cards free on two reads 30 s apart and gpu0..3.lock free. Hard stop: HARD_S (900 s) after the start; every step is
# killed at the deadline. At the end releases gpu0..3.lock and touches dsv41.l30-4card-done.
# CPUs 24-31,56-63. Start: (cd /home/access/dsv41 && nohup setsid bash spec/scripts/w14_l30.sh) > spec/logs/w14.log 2>&1 < /dev/null &
set -u
D=/home/access/dsv41
EV=/mnt/Data/work/evals
P=$D/perf-bins
S=$D/spec
R=$D/receipts-perf/w14-l30
A=l30main; B=l30a1337; H=l30b1343
HARD_S=${HARD_S:-900}
mkdir -p $R
cd $D || exit 1
say() { echo "$(date +%T) $*" | tee -a $R/summary.txt; }
finish() { touch $EV/dsv41.l30-4card-done; say "touched dsv41.l30-4card-done"; }
for t in $A $B $H; do [ -x $P/$t/dsv41-tpbench ] && [ -x $P/$t/dsv41-gate ] || { say "missing $P/$t"; exit 2; }; say "$t $(cat $P/$t/SOURCE)"; done
touch $S/logs/w14.armed
touch -d "$(date +%F) 02:30" $S/logs/w14.after
gu=$(date -d "${GIVE_UP:-tomorrow 06:00}" +%s)
open() { [ -e $EV/dssparks.window-0123 ] && [ $EV/dssparks.window-0123 -nt $S/logs/w14.armed ] && [ $EV/dssparks.window-0123 -nt $S/logs/w14.after ]; }
free4() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
say "armed; waiting for dssparks.window-0123; give up $(date -d @$gu '+%F %H:%M')"
while :; do
  [ "$(date +%s)" -lt $gu ] || { say "no window by the give-up time: not run"; exit 3; }
  [ -e /mnt/Data/work/memra-the box-ds/QUIET ] && { say "QUIET set: not run"; exit 3; }
  if open && free4; then sleep 30; free4 && break; fi
  sleep 10
done
for i in 0 1 2 3; do
  exec {fd}>>"$EV/gpu$i.lock"
  if ! flock -n "$fd"; then say "ABORT: gpu$i.lock is held by someone else; nothing run"; finish; exit 2; fi
  eval "FD$i=$fd"
done
export TP4_HELD=1
t0=$(date +%s); dl=$((t0 + HARD_S))
left() { echo $((dl - $(date +%s))); }
say "W14-START: window from ds-sparks; all four box locks held; host free $(free -g | awk '/Mem:/{print $7}') GiB; load $(cut -d' ' -f1-3 /proc/loadavg); hard stop $(date -d @$dl +%T)"
nvidia-smi --query-gpu=index,name,power.limit,temperature.gpu,clocks.sm --format=csv,noheader >> $R/summary.txt

# 1. #1337: kernel gate, three tp gates, timed A/B (budget: what is left minus the #1343 phases' 5 minutes)
b1=$(( $(left) - 300 )); [ $b1 -gt 660 ] && b1=660
env K3_ARM_A=$A K3_ARM_B=$B K3_TAG=l30-1337-four K3_CARDS=0123 K3_DEV=0,1,2,3 K3_LAYERS=40 \
  K3_ROWS="4 8 9 10 12 16" K3_ORDER="A B B A B A A B" K3_KGATE=1 K3_GATE_T="default peer-pull host-canonical" \
  K3_CPUS=24-31,56-63 timeout -k 10 $(( $(left) - 5 )) bash bin/perf_cfix_four.sh $b1 > $R/l30-1337-four.out 2>&1
say "#1337 $B vs $A (four cards) rc $?; load $(cut -d' ' -f1-3 /proc/loadavg)"
grep -E "kgate:|gate-(default|peer-pull|host-canonical):|decode tokens|flagged|skipped" $R/l30-1337-four.out | tee -a $R/summary.txt

# 2. main's kernel gate (untimed, card 0)
mkdir -p $R/kgate-main
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MEMRA_FULL_PREC=1 taskset -c 24-31,56-63 timeout -k 5 \
  $(( $(left) > 60 ? 60 : $(left) )) $P/$A/dsv41-gate kernels $D/fixtures-gpu/rand --out $R/kgate-main/kgate.json \
  > $R/kgate-main/kgate.log 2>&1
say "main kgate rc $?: $(grep -a -o '[0-9]* checks, [0-9]* failed' $R/kgate-main/kgate.log | tail -1)"

# 3. #1343's battery: kgate and tp4, BASE = #1337's binaries, HEAD = #1343's (symlinked where the script looks)
O=$R/bs1343
for a in BASE HEAD; do mkdir -p $O/target-$a/release; done
t=$B; for a in BASE HEAD; do for b in dsv41-gate; do ln -sf $P/$t/$b $O/target-$a/release/$b; done; t=$H; done
for ph in kgate tp4; do
  l=$(left); [ $l -gt 20 ] || { say "#1343 $ph: not run, hard stop reached"; continue; }
  env OUT=$O SRC_BASE=$D/wb/src-l30a1337-2df9c40b39 SRC_HEAD=$D/wb/src-l30b1343-868417afce DATA=$D \
    CKPT=$D/DeepSeek-V4.1-Flash-1017 CARD1=0 CARDS4=0,1,2,3 MEMRA_GPU_LOCK=$R/box.lock \
    taskset -c 24-31,56-63 timeout -k 10 $l bash $S/scripts/gpu-battery-1343.sh $ph > $R/bs1343-$ph.out 2>&1
  say "#1343 $ph rc $?: $(tail -n 1 $R/bs1343-$ph.out)"
done
grep -E "FAIL|equal|kgate (BASE|HEAD)|tp4 " $O/battery.log | tee -a $R/summary.txt

for i in 0 1 2 3; do eval "fd=\$FD$i"; flock -u "$fd"; eval "exec $fd>&-"; done
say "all four box locks released ($(( $(date +%s) - t0 )) s in the window)"
finish
say W14-DONE
