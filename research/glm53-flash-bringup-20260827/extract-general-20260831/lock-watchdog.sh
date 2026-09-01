#!/usr/bin/env bash
# Rig-lock watchdog (extract-general lane, 2026-08-31): the sccache daemon inherits the
# /tmp/memra-5090.lock fd from flock'd cargo builds; when the flock holder exits, the
# exclusive lock lives on in the daemon and /proc/locks names a DEAD pid as holder —
# every queued lane deadlocks (measured: ~80 min of idle 5090 across three lanes today).
# This watchdog releases ONLY that poisoned state: lock held by a pid that no longer
# exists AND an sccache daemon holding the lock-file fd -> kill the daemon (it restarts
# on demand; losing warm cache state is noise next to a deadlocked GPU).
set -u
LOCK=/tmp/memra-5090.lock
INODE=$(stat -c %i "$LOCK")
while true; do
  line=$(grep " 00:30:$INODE " /proc/locks | grep -v -- '->' | head -1 || true)
  if [ -n "$line" ]; then
    holder=$(echo "$line" | awk '{print $5}')
    if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
      for p in $(pgrep -x sccache); do
        if ls -l "/proc/$p/fd" 2>/dev/null | grep -q "memra-5090.lock"; then
          echo "$(date -Is) watchdog: lock holder $holder is dead; killing fd-holding sccache $p"
          kill "$p" 2>/dev/null
        fi
      done
    fi
  fi
  sleep 20
done
