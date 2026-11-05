#!/usr/bin/env bash
# DAY49 addendum D serving-shape boots: day49-run.sh's boots with the fault aimed (FAULT, default batch:1) and the route
# per arm. One boot per spec through run-day26-cell.sh (it takes the rig lock), after a bounded idle wait (at most
# 7200 s: lock free, no compute app, >= 24 GB host memory). Never a signal to anything this lane did not start.
# usage: day49d-run.sh <receipt root> <name>:<arm> ...
#   arm = off | on (MEMRA_BATCH_OOM_RECOVER=1) on the spec-default route; off-plain | on-plain the same with
#   MEMRA_SERVE_SPEC=0. Every arm sets MEMRA_STEP_OOM_FAULT=$FAULT. day45-client.py's warm, burst and probe; env
#   CLIENT_EXTRA passes the burst shape.
# env: WT, RIG_LOCK (/tmp/memra-5090.lock), BIN (required), FAULT, MODEL, MODEL_KEY, BOOT_CTX (65536; empty = the
#      checkpoint's), NO_SCOPE, YIELD_S.
#
# Max-Q kit port (kit/DEVIATIONS.md D6-D8), the only changes against the copy at 2ccb038d1c (kit/orig/, byte-identical
# to the Workstation chain tree 6cef1bbe3's):
# - The arms switch BINARIES, not the env door. MEMRA_BATCH_OOM_RECOVER is retired at the kit's source and the boot
#   audit refuses it, so `on`/`on-plain` run BIN_ON (2ccb038d1c plus maxq-batch-oom-recover.patch: the recovery is the
#   Max-Q default) and `off`/`off-plain` run BIN_OFF (2ccb038d1c: the Max-Q keeps the error arm). The route env, the
#   fault, the client shape and the boot order are the registered ones.
# - idle() reads the pinned card only (`nvidia-smi -i $CARD`): the other cards of the box train. The lock probe is
#   dropped: the kit's run.sh holds /tmp/memra-gpu.lock for the whole cell and passes LOCK=none to the cell, as the
#   collector does. The idle wait is IDLE_WAIT_S (default 300 s), not 7200 s: inside the window the card is ours.
# - Each boot records its binary label and source in arm.txt and hands them to run-day26-cell.sh (no git on <box>).
set -uo pipefail
R=${1:?receipt root}; shift
WT=${WT:?WT: the kit root}
CARD=${CARD:?CARD: the nvidia-smi index of the pinned card}
BIN_OFF=${BIN_OFF:?the off binary} BIN_ON=${BIN_ON:?the on binary}
SRC_OFF=${SRC_OFF:?the off binary source} SRC_ON=${SRC_ON:?the on binary source}
FAULT=${FAULT:-batch:1}
LENGTHS=${LENGTHS:-6144,30720}
IDLE_WAIT_S=${IDLE_WAIT_S:-300}
cd "$WT" || exit 1
mkdir -p "$R/boots"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/run.log"; }
idle() {
  [ -z "$(nvidia-smi -i "$CARD" --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
BOOT_CTX=${BOOT_CTX-65536}
if [ -n "$BOOT_CTX" ]; then export MEMRA_CTX=$BOOT_CTX; else unset MEMRA_CTX; fi
export LOCK=none RIGDIR="$R/boots" N=5 CLIENT=day45-client.py PARSER=day45-parse.py MEMRA_TIMEOUT_MS_MAX=3600000
for spec in "$@"; do
  IFS=: read -r name arm <<< "$spec"
  uenv=(-u MEMRA_BATCH_OOM_RECOVER -u MEMRA_STEP_OOM_FAULT -u MEMRA_ADMIT_W_RELEASE -u MEMRA_ADMIT_PREDICT_SHADOW -u MEMRA_ADMIT_BY_MEMORY -u MEMRA_SERVE_SPEC)
  aenv=(MEMRA_STEP_OOM_FAULT="$FAULT")
  case $arm in
    off) B=$BIN_OFF; label=off; src=$SRC_OFF ;;
    on) B=$BIN_ON; label=on; src=$SRC_ON ;;
    off-plain) B=$BIN_OFF; label=off; src=$SRC_OFF; aenv+=(MEMRA_SERVE_SPEC=0) ;;
    on-plain) B=$BIN_ON; label=on; src=$SRC_ON; aenv+=(MEMRA_SERVE_SPEC=0) ;;
    *) log "boot $name: unknown arm $arm"; exit 1 ;;
  esac
  args="${CLIENT_EXTRA:-}"
  [ -x "$B" ] || { log "boot $name: no binary $B; not run"; exit 1; }
  deadline=$((SECONDS + IDLE_WAIT_S)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "boot $name: card $CARD not idle after $IDLE_WAIT_S s; not run"; exit 3; }
    [ $waited = 0 ] && log "boot $name: waiting for an idle card $CARD"
    waited=1; sleep 10
  done
  log "boot $name start arm=$arm bin=$(sha256sum "$B" | cut -c1-16) label=$label env=[${aenv[*]}] args=[$args]"
  printf 'arm=%s\nbin_label=%s\nbin_sha256=%s\nsource=%s\nenv=%s\nclient_args=%s\n' "$arm" "$label" \
    "$(sha256sum "$B" | cut -d' ' -f1)" "$src" "${aenv[*]}" "$args" > "$R/boots/$name.arm.txt"
  env "${uenv[@]}" "${aenv[@]}" CLIENT_ARGS="$args" CELL_SOURCE="$src" bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$B" \
    > "$R/boots/$name.launch.log" 2>&1
  log "boot $name rc=$? $(tail -1 "$R/boots/$name/REPORT.txt" 2>/dev/null | cut -c1-160)"
  # The runner's thermal guard or deadline stops the boots between cells (kit/run.sh).
  [ -e "$R/STOP" ] && { log "stop requested: $(head -1 "$R/STOP")"; exit 4; }
  sleep "${YIELD_S:-5}" # the lane yields the card between cells when YIELD_S is set (lead, 2026-09-26)
done
log "boots done: $*"
