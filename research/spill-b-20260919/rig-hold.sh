# Sourced by the lane's boot runners (DAY48 addendum D, 2026-09-27): hold the rig lock first, then check the rig idle
# under the hold, then boot. A lock taker that re-takes the lock per run, back to back, cannot starve a boot: a blocked
# flock waiter is woken at each release, where a flock -n poll only sees the lock free in the gap between two runs.
# A rig still busy under the hold (a process that does not take the lock) is waited on under the hold for at most
# HOLD_IDLE_S (120 s); then the hold is released and retaken after HOLD_RETRY_S (30 s), within the boot's deadline.
# needs: RIG_LOCK and log() from the runner. fd 8 carries the hold; run the cell with 8>&- so no child inherits it.
# env: MIN_HOST_GB (24), HOLD_IDLE_S (120), HOLD_RETRY_S (30), HOLD_POLL_S (10), SMI_TIMEOUT_S (30).

# Prints why the rig is busy; prints nothing when it is idle (1.2's conditions: no compute app, host memory available).
rig_busy() {
  local apps mem
  if ! apps=$(timeout "${SMI_TIMEOUT_S:-30}" nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>&1); then
    echo "nvidia-smi failed or timed out: $(echo "$apps" | tr '\n' ' ' | cut -c1-120)"
    return
  fi
  [ -z "$apps" ] || { echo "compute apps: $(echo "$apps" | tr '\n' ';' | cut -c1-200)"; return; }
  mem=$(free -g | awk '/^Mem:/{print $7}')
  [ "$mem" -ge "${MIN_HOST_GB:-24}" ] || echo "host memory available ${mem} GB < ${MIN_HOST_GB:-24} GB"
}

# rig_hold <boot name> <deadline, in bash SECONDS>: 0 = the lock is held and the rig idle under the hold; 3 = the deadline
# passed first (the hold is not kept).
rig_hold() {
  local name=$1 deadline=$2 left t0 why busy_until
  exec 8>"$RIG_LOCK"
  while :; do
    left=$((deadline - SECONDS))
    if [ "$left" -le 0 ]; then
      log "boot $name: deadline passed before the rig was held idle"
      exec 8>&-
      return 3
    fi
    t0=$SECONDS
    if ! flock -w "$left" 8; then
      log "boot $name: $RIG_LOCK not acquired before the deadline"
      exec 8>&-
      return 3
    fi
    log "boot $name: $RIG_LOCK held after $((SECONDS - t0)) s"
    busy_until=$((SECONDS + ${HOLD_IDLE_S:-120}))
    why=$(rig_busy)
    while [ -n "$why" ] && [ "$SECONDS" -lt "$busy_until" ] && [ "$SECONDS" -lt "$deadline" ]; do
      sleep "${HOLD_POLL_S:-10}"
      why=$(rig_busy)
    done
    if [ -z "$why" ]; then
      log "boot $name: idle under the hold"
      return 0
    fi
    log "boot $name: busy under the hold ($why); hold released, retry in ${HOLD_RETRY_S:-30} s"
    flock -u 8
    sleep "${HOLD_RETRY_S:-30}"
  done
}

# rig_release <boot name>: drops the hold after the cell.
rig_release() {
  flock -u 8 2>/dev/null
  exec 8>&-
  log "boot $1: $RIG_LOCK released"
}
