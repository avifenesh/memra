#!/usr/bin/env bash
# WP-B local 5090 queue-n (2026-09-27): the boots two runners left unrun when their idle wait ran out while another lane
# held the card (their run.logs say `rig not idle after 7200 s; not run`): DAY46C's O1-enforce-wrel, O2-enforce-wrel,
# O2-enforce and O2-shadow, and DAY49D's O2-off, P1-off, P1-on, P2-on and P2-off. Same runners, specs, binaries and
# receipt roots; a spec whose `boot <name> start` line is in its run.log ran and is never asked again, and a call that
# exits 3 is asked again with the specs not yet started, up to eight times (queue-m's rule). Runs from the lane's integ71
# tree. Builds at nice 19 under 600%. Never a signal to anything.
set -uo pipefail
export WT=$HOME/projects/wt-b-integ71
D=$WT/research/spill-b-20260919
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-n.log"; }
started() { command grep -q "boot $2 start" "$1/run.log" 2>/dev/null; }
ask() { # <runner> <R> <spec>...
  local runner=$1 R=$2; shift 2
  local try spec rc left
  for try in 1 2 3 4 5 6 7 8; do
    left=(); for spec in "$@"; do started "$R" "${spec%%:*}" || left+=("$spec"); done
    [ ${#left[@]} = 0 ] && return 0
    [ $try = 1 ] || log "$(basename "$runner") ${left[*]}: asked again (try $try) after an idle-wait timeout"
    bash "$D/$runner" "$R" "${left[@]}"; rc=$?
    [ $rc = 3 ] || return $rc
  done
  log "$(basename "$runner") ${left[*]}: not run after eight idle waits"; return 3
}
log "queue-n start HEAD=$(git rev-parse HEAD)"
# 1. DAY46C (the binary DAY46's local half built at 8926ccfb3).
R=$D/rtx5090-day46c
( export BIN=$HOME/projects/wt-spill-b/target/day46/tip/memra-server YIELD_S=240 \
    CLIENT_EXTRA="--burst 32 --length 6144 --max-tokens 64"
  ask day46-run.sh "$R" O1-shadow:shadow O1-enforce:enforce O1-enforce-wrel:enforce-wrel \
    O2-enforce-wrel:enforce-wrel O2-enforce:enforce O2-shadow:shadow )
log "day 46c boots rc=$?"
python3 "$D/day46-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
# 2. DAY49D's serving boots (green rebuilt at 8926ccfb3 by build-arms.sh; the gates already ran).
R=$D/rtx5090-day49d
if [ ! -s "$WT/target/day49d/SHA256SUMS" ]; then
  echo "$(date -u +%FT%TZ) WP-B queue-n DAY49D green build (nice 19, CPUQuota=600%)" >> "$D/cpu-concurrency.log"
  TARGET=$WT/target/arms-build WRAP="systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19" \
    bash "$D/build-arms.sh" "$WT/target/day49d" 8926ccfb3e8a27cf8f91aa747e2b088251215c7a tip > "$D/rtx5090-queue-n-day49d-build.out" 2>&1
  rc=$?
  log "day 49d green build rc=$rc"
fi
( export BIN=$WT/target/day49d/tip/memra-server FAULT=batch:1 YIELD_S=240 \
    CLIENT_EXTRA="--warm-n 0 --burst 8 --length 6144 --max-tokens 64"
  ask day49d-run.sh "$R" O1-off:off O1-on:on O2-on:on O2-off:off P1-off:off-plain P1-on:on-plain P2-on:on-plain P2-off:off-plain )
log "day 49d boots rc=$?"
python3 "$D/day49d-read.py" serve rtx5090 "$R" > "$R/read-serve.log" 2>&1
log "queue-n done"
