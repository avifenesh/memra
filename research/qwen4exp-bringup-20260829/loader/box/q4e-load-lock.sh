#!/usr/bin/env bash
# q4e-load-lock.sh <logfile> <cmd...>
# Box-wide LOAD serializer for the 4-card lane box (353 GB host). The qwen4_exp loader stages
# the whole 174 GB artifact in host RAM before uploading (anon-RSS ~180 GB at peak), so two
# concurrent loads exceed the host and the GLOBAL OOM killer takes one of them — receipted
# 2026-09-02 00:15:09Z (pid 28271, trace re-run, anon-rss 180.8 GB, constraint=CONSTRAINT_NONE)
# while two other lanes were loading. The measurement lock does not prevent this: shared
# holders load concurrently. This wrapper:
#   1. takes an EXCLUSIVE load lock,
#   2. waits until host MemAvailable >= LOAD_NEED_GB (default 200),
#   3. starts the command with stdout+stderr into <logfile>,
#   4. holds the load lock until the binary prints its "post-load" vram line (or exits),
#   5. releases the load lock and waits for the command, propagating its exit code, and
#      appends an explicit "# load-lock rc=<n> killed=<yes/no>" line so a SIGKILL can never
#      read as rc=0 in a queue log.
set -u
LOG=$1; shift
LOCK=/tmp/q48fn-load.lock
NEED=${LOAD_NEED_GB:-200}
exec 9>"$LOCK"; flock -x 9
while [ "$(free -g | awk 'NR==2{print $7}')" -lt "$NEED" ]; do sleep 20; done
"$@" > "$LOG" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null && ! grep -q "post-load" "$LOG" 2>/dev/null; do sleep 5; done
exec 9>&-
wait "$pid"; rc=$?
killed=no; [ "$rc" -ge 128 ] && killed=yes
printf '# load-lock\trc=%s\tkilled=%s\n' "$rc" "$killed" >> "$LOG"
exit "$rc"
