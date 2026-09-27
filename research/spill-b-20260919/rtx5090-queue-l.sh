#!/usr/bin/env bash
# WP-B local 5090 queue-l (2026-09-26): one serial runner for the lane's gate cells. It replaces three runners that each
# waited for the card on its own (rtx5090-day47/run.sh with RUNS="a2 a3", rtx5090-day49/run.sh, rtx5090-day49/run-c.sh),
# stopped by the lane in their idle waits with nothing run (each run.log says so). Same cells, arms, env and receipt
# paths. What changes is how the lock is taken: after the idle check this runner takes /tmp/memra-5090.lock itself
# (fd 9) and runs the gate under it; a lock not acquired within 60 s sends the unit back to the idle wait instead of
# recording a run with no gate output (the old wrappers' `flock -w 600` timeout exit, which two of the lane's runners
# polling the same free window would hit). The lock stays free for 240 s after every unit. Never a signal.
# Order: DAY47 a2, a3 (the lane checkout, as rtx5090-day47/run.sh ran); DAY49 g1, g2 (the 02dbdfa40 worktree, the
# registered source); DAY49C green (95d35c383) and red (02dbdfa40) under MEMRA_KV_ALLOCATOR=vmm, then its reader; the
# DAY49 serving boots (target/day49/tip); then the two worktrees are removed.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
LOCK=/tmp/memra-5090.lock
GREEN=$WT/target/wt-day49c-green
RED=$WT/target/wt-day49c-red
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-l.log"; }
idle() { flock -n "$LOCK" true && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
gate_unit() { # <run log> <label> <tree> <out dir> <env...>: <tree>/tools/health-fault-gate.sh under the lock
  local runlog=$1 label=$2 tree=$3 out=$4; shift 4
  local deadline=$((SECONDS + 28800)) rc
  while :; do
    until idle; do
      [ $SECONDS -ge $deadline ] && { echo "$(date -u +%FT%TZ) $label: card not idle after 28800 s; not run" >> "$runlog"; return 3; }
      sleep 30
    done
    exec 9>"$LOCK"
    flock -w 60 9 && break
    exec 9>&-
    log "$label: lock taken by another runner inside the window; back to the idle wait"
  done
  echo "$(date -u +%FT%TZ) $label start HEAD=$(git -C "$tree" rev-parse HEAD)" >> "$runlog"
  env "$@" HFG_OUT="$out" "$tree/tools/health-fault-gate.sh" > "$out.log" 2>&1
  rc=$?
  exec 9>&-
  echo "$(date -u +%FT%TZ) $label rc=$rc $(tail -1 "$out.log")" >> "$runlog"
  sleep 240
  return $rc
}
[ -x "$GREEN/target/release/memra-server" ] && [ -x "$RED/target/release/memra-server" ] || { log "worktree binaries missing"; exit 2; }
log "queue-l start HEAD=$(git rev-parse HEAD) green=$(git -C "$GREEN" rev-parse --short HEAD) red=$(git -C "$RED" rev-parse --short HEAD)"
# 1. DAY47 addendum B's reruns (arms g and h).
R47=$D/rtx5090-day47
for run in a2 a3; do gate_unit "$R47/run.log" "$run" "$WT" "$R47/$run" HFG_ARMS=g,h; done
log "day 47 a2 a3 done"
# 2. DAY49's gate arm i twice, on the registered source.
R49=$D/rtx5090-day49
for run in g1 g2; do gate_unit "$R49/run.log" "gate $run" "$RED" "$R49/$run" HFG_ARMS=i; done
log "day 49 g1 g2 done"
# 3. DAY49 addendum C's i-vmm cell.
R49C=$D/rtx5090-day49c
gate_unit "$R49C/run.log" green "$GREEN" "$R49C/green" MEMRA_KV_ALLOCATOR=vmm HFG_ARMS=i
gate_unit "$R49C/run.log" red "$RED" "$R49C/red" MEMRA_KV_ALLOCATOR=vmm HFG_ARMS=i
python3 "$D/day49c-read.py" rtx5090 "$R49C/green" "$R49C/red" > "$R49C/read.log" 2>&1
log "day 49c done"
# 4. DAY49's serving shape (day49-run.sh waits for the card and takes the lock per boot itself).
BIN=$WT/target/day49/tip/memra-server YIELD_S=240 CLIENT_EXTRA="--warm-n 0 --burst 8 --length 6144 --max-tokens 64" \
  bash "$D/day49-run.sh" "$R49" O1-off:off O1-on:on O2-on:on O2-off:off
echo "$(date -u +%FT%TZ) boots rc=$?" >> "$R49/run.log"
git worktree remove --force "$GREEN" >> "$D/rtx5090-queue-l.log" 2>&1
git worktree remove --force "$RED" >> "$D/rtx5090-queue-l.log" 2>&1
log "queue-l done"
