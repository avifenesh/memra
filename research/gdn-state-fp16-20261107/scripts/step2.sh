#!/usr/bin/env bash
# Rig-side driver for step 2 of research/gdn-state-fp16-20261107 on one GB10 unit, in order:
# build (lane bins + GPU tests), build main's bins (identity baseline), gates, the rest of the
# census (131K + 258K cold primes), the quality panels fp16 / fp32 control / fp16, the decode tick
# bench, TTFT A B B A. Each step leaves runs/gb10/<unit>/<step>.done; a rerun skips finished steps.
# A failed gate stops the chain before any panel.
#   step2.sh UNIT LEASE [STEPS...]   (default: every step)
set -u
UNIT=${1:?unit}; LEASE=${2:?lease}; shift 2
L=$HOME/.cache/memra-gdnfp16
C=$L/scripts/chain.sh
O=$L/runs/gb10/$UNIT
ARM=arm-$(git -C "$L/wt-arm" log -1 --format=%H -- crates | cut -c1-10)
MB=main-$(git -C "$L/wt-arm" merge-base HEAD origin/main | cut -c1-10)
mkdir -p "$O"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$O/step2.log"; }
steps=${*:-build buildmain gates census-long q1 q0 q2 bench ttft}
log "step2 unit=$UNIT arm=$ARM main=$MB steps: $steps"
for s in $steps; do
  [ -e "$O/$s.done" ] && { log "skip $s (done)"; continue; }
  log "start $s"
  case $s in
    build) WHAT=all JOBS=6 bash "$C" "$UNIT" "$LEASE" "$ARM" build "$O/build" ;;
    buildmain) JOBS=6 bash "$C" "$UNIT" "$LEASE" "$MB" buildmain "$O/buildmain" ;;
    gates)
      MAINBIN=\$HOME/.cache/memra-gdnfp16/bins/$MB bash "$C" "$UNIT" "$LEASE" "$ARM" gates "$O/gates"
      g=$O/gates/gates/gates.log
      id_main=$(grep -oE 'ident main pp_forward_last=[0-9a-f]+ numeric=[0-9a-f]+' "$g" | cut -d' ' -f3-)
      id_new0=$(grep -oE 'ident new0 pp_forward_last=[0-9a-f]+ numeric=[0-9a-f]+' "$g" | cut -d' ' -f3-)
      if [ ! -s "$g" ] || grep -qE '[1-9][0-9]* FAIL|rc=[1-9]|FAILED' "$g" || [ -z "$id_main" ] || [ "$id_main" != "$id_new0" ]; then
        log "gates not clean (main: $id_main, new0: $id_new0): stopping"; cat "$g" >> "$O/step2.log"; exit 3
      fi ;;
    census-long) bash "$C" "$UNIT" "$LEASE" "$ARM" census-long "$O/census-long" ;;
    q1) QCLASS=1 QREP=1 bash "$C" "$UNIT" "$LEASE" "$ARM" quality "$O/quality" ;;
    q0) QCLASS=0 QREP=1 bash "$C" "$UNIT" "$LEASE" "$ARM" quality "$O/quality" ;;
    q2) QCLASS=1 QREP=2 bash "$C" "$UNIT" "$LEASE" "$ARM" quality "$O/quality" ;;
    bench) bash "$C" "$UNIT" "$LEASE" "$ARM" bench "$O/bench" ;;
    ttft) bash "$C" "$UNIT" "$LEASE" "$ARM" ttft "$O/ttft" ;;
    *) log "unknown step $s"; exit 2 ;;
  esac
  rc=$?
  log "end $s rc=$rc"
  [ $rc = 0 ] && touch "$O/$s.done"
done
log STEP2_DONE
