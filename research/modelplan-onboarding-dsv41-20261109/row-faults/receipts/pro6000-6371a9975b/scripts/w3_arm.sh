#!/bin/bash
# wrx, window 3 (2026-11-14): memra #1311's fix on four cards, then PR #1315's PRO 6000 checks on cards 2+3.
# agy's protocol: dsv41.w3-ready touched by us; agy writes a fresh gpu01.dsv41-window-open with gpu0..3.lock free; this
# runner starts when that file is newer than its own start and all four cards are free on two reads 30 s apart, and
# touches gpu01.dsv41-window-done when our part ends. It never touches gpu23.* markers.
#   phase A (cards 0123, 40 layers, quiet box): perf_cfix_four.sh main24 (A) against gather1 (B), rows 4 8 9 10 12 16,
#     A B B A B A A B, kernel gate + tp gates of B first; then the pre-CUTLASS base route2 (A) against gather1, no gates.
#     Then touches dsv41.w3-4card-done: cards 0+1 are free (ds-sparks' two-card run).
#   phase B (cards 2+3; binaries perf-bins/land-1315, tree src-m1 = the same commit):
#     a. rowfault over all 40 layers on the one-card residency (card 2)
#     b. the serve smoke at 40 layers on the natural one-card residency (card 2; spec/scripts/smoke_wrx.sh)
#     d. rowfault on the pair: 2 layers resident, 2 layers --experts-host, 40 layers --experts-host
#     e. M1: verify checkpoint + verify serve (card 2; spec/scripts/m1_verify.sh)
# env PHASES (default "A B"), GIVE_UP (default today 23:30). Markers W3-START, W3-A-DONE, W3-DONE in spec/logs/w3.log.
# Start: (cd /home/access/dsv41 && nohup setsid bash spec/scripts/w3_arm.sh) > spec/logs/w3.log 2>&1 < /dev/null &
set -u
D=/home/access/dsv41
EV=/mnt/Data/work/evals
C=$D/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
P=$D/perf-bins
S=$D/spec
R=$D/receipts-perf/w3-1114
T15=$P/${TAG1315:-land-1315}
mkdir -p $R $D/rowfault-1315
cd $D || exit 1
say() { echo "$(date +%T) $*" | tee -a $R/summary.txt; }
res() { say "$1: $(grep -o '[0-9]* checks, [0-9]* failed' $2 | tail -1) $(grep -ac '^FAIL' $2) FAIL lines"; }
touch $S/logs/w3.armed
gu=$(date -d "${GIVE_UP:-23:30}" +%s)
open() { [ -e $EV/gpu01.dsv41-window-open ] && [ $EV/gpu01.dsv41-window-open -nt $S/logs/w3.armed ]; }
free4() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
case " ${PHASES:-A B} " in *" A "*) for t in main24 gather1 route2; do [ -x $P/$t/dsv41-tpbench ] || { say "missing $P/$t"; exit 2; }; done ;; esac
case " ${PHASES:-A B} " in *" B "*) for b in dsv41-gate memra-server memra dsv41-run; do [ -x $T15/$b ] || { say "missing $T15/$b"; exit 2; }; done ;; esac
say "waiting for agy's window-open; give up $(date -d @$gu +%H:%M)"
while :; do
  [ "$(date +%s)" -lt $gu ] || { say "no window by the give-up time: not run"; exit 3; }
  [ -e /mnt/Data/work/memra-wrx-ds/QUIET ] && { say "QUIET set: not run"; exit 3; }
  if open && free4; then sleep 30; free4 && break; fi
  sleep 20
done
# agy's protocol (inbox 20261114-0620): hold every card's box lock from the claim; at the end of part A keep gpu2.lock
# and gpu3.lock, unlock gpu0.lock and gpu1.lock, touch dsv41.w3-4card-done (ds-sparks starts on 0+1); at the end unlock
# gpu2/gpu3 and touch gpu01.dsv41-window-done. TP4_HELD=1: the leases below take only the card leases (every step here
# goes through perf_runc.sh -> tp4_lease.sh; none uses the one-card gpu0q/lease.sh path that would wait on a held lock).
for i in 0 1 2 3; do
  exec {fd}>>"$EV/gpu$i.lock"
  if ! flock -n "$fd"; then
    # the window is open but a lock is taken: run nothing, hand the window back so the chain (ds-sparks on 0+1, layby on
    # 2+3, agy's controller) is not left waiting on markers only we write. Locks taken so far drop with this process.
    say "ABORT: gpu$i.lock is held by someone else; nothing run"
    touch $EV/dsv41.w3-4card-done $EV/gpu01.dsv41-window-done
    say "touched dsv41.w3-4card-done and gpu01.dsv41-window-done (abort)"
    exit 2
  fi
  eval "FD$i=$fd"
done
export TP4_HELD=1
unlock() { for i in "$@"; do eval "fd=\$FD$i"; flock -u "$fd"; eval "exec $fd>&-"; done; }
say "W3-START: $(tr '\n' ' ' < $EV/gpu01.dsv41-window-open); all four box locks held; host free $(free -g | awk '/Mem:/{print $7}') GiB"
for ph in ${PHASES:-A B}; do
  case $ph in
  A)
    env K3_ARM_A=main24 K3_ARM_B=gather1 K3_TAG=gather1-four K3_CARDS=0123 K3_DEV=0,1,2,3 K3_LAYERS=40 \
      K3_ROWS="4 8 9 10 12 16" K3_ORDER="A B B A B A A B" K3_KGATE=1 K3_GATE_T="default peer-pull host-canonical" \
      bash bin/perf_cfix_four.sh 840 > $R/gather1-four.out 2>&1
    say "gather1 vs main24 (four cards) rc $?"
    env K3_ARM_A=route2 K3_ARM_B=gather1 K3_TAG=gather1-base-four K3_CARDS=0123 K3_DEV=0,1,2,3 K3_LAYERS=40 \
      K3_ROWS="4 8 9 10 12 16" K3_ORDER="A B B A B A A B" K3_GATES=0 bash bin/perf_cfix_four.sh 660 > $R/gather1-base-four.out 2>&1
    say "gather1 vs route2 (four cards) rc $?"
    unlock 0 1; A_DONE=1
    touch $EV/dsv41.w3-4card-done
    say "W3-A-DONE: gpu0.lock and gpu1.lock released, dsv41.w3-4card-done touched (gpu2/gpu3 still held)"
    ;;
  B)
    say "land-1315 $(cat $T15/SOURCE)"
    O=$D/rowfault-1315
    bash bin/perf_runc.sh w3-1114/rowfault-l40-host 900 2 env MEMRA_FULL_PREC=1 $T15/dsv41-gate rowfault $C --token-map $TM \
      --layers 40 --experts-host --out $O/rowfault-l40-host.json > $O/rowfault-l40-host.log 2>&1
    say "(a) rowfault 40 layers one-card residency rc $?"; res "(a)" $O/rowfault-l40-host.log
    bash bin/perf_runc.sh w3-1114/smoke 1200 2 bash $S/scripts/smoke_wrx.sh ${TAG1315:-land-1315} $O/smoke > $O/smoke.out 2>&1
    say "(b) serve smoke 40 layers rc $? $(grep -c '^PASS' $O/smoke/s.txt 2>/dev/null) PASS lines, $(grep -c '^FAIL' $O/smoke/s.txt 2>/dev/null) FAIL; $(tail -n 2 $O/smoke/s.txt 2>/dev/null | tr '\n' ' ')"
    for a in "pair-resident:--layers 2" "pair-host:--layers 2 --experts-host" "pair-l40-host:--layers 40 --experts-host"; do
      n=${a%%:*}; f=${a#*:}
      bash bin/perf_runc.sh w3-1114/rowfault-$n 900 23 env MEMRA_FULL_PREC=1 $T15/dsv41-gate rowfault $C --token-map $TM \
        $f --devices 0,1 --out $O/rowfault-$n.json > $O/rowfault-$n.log 2>&1
      say "(d) rowfault $n rc $?"; res "(d) $n" $O/rowfault-$n.log
    done
    M1_TAG=${TAG1315:-land-1315} bash bin/perf_runc.sh w3-1114/m1-verify 2400 2 bash $S/scripts/m1_verify.sh > $R/m1-verify.out 2>&1
    say "(e) M1 lease rc $?"; grep -E 'checkpoint rc|serve rc|=passed|=pending|=failed|M1-DONE|differ' $R/m1-verify.out | tee -a $R/summary.txt
    ;;
  esac
done
[ "${A_DONE:-0}" = 1 ] || { unlock 0 1; touch $EV/dsv41.w3-4card-done; say "no phase A: gpu0/gpu1 released, dsv41.w3-4card-done touched"; }
unlock 2 3
touch $EV/gpu01.dsv41-window-done
say "gpu2.lock and gpu3.lock released, gpu01.dsv41-window-done touched"
say W3-DONE
