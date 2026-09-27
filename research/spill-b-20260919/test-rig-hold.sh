#!/usr/bin/env bash
# DAY48 addendum D: checks rig-hold.sh against a fixture lock in a scratch dir (it guards nothing; no GPU lock is touched)
# and stub nvidia-smi and free on PATH. Prints one PASS/FAIL line per check and a final line; exit 0 only when all pass.
#   T1 a lock taker that re-takes the lock back to back: the hold gets in; the old flock -n poll's hits are a reading.
#   T2 a compute app that clears under the hold: held, idle under the hold, no release.
#   T3 a compute app that never clears: busy under the hold, released between tries, the deadline ends it (rc 3).
#   T4 the hold is real (another flock -n fails) and a child run with 8>&- does not inherit fd 8 (one without does).
#   T5 rig_release frees the lock.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d /tmp/b-rig-hold-test.XXXXXX)
cleanup() { touch "$T/stop"; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
mkdir -p "$T/bin"
cat > "$T/bin/nvidia-smi" <<EOF
#!/usr/bin/env bash
[ -e "$T/app" ] && echo "4242, fake-app, 1000 MiB"
exit 0
EOF
cat > "$T/bin/free" <<'EOF'
#!/usr/bin/env bash
echo "              total        used        free      shared  buff/cache   available"
echo "Mem:             64          10          30           0          24          50"
EOF
chmod +x "$T/bin/nvidia-smi" "$T/bin/free"
export PATH="$T/bin:$PATH"
RIG_LOCK=$T/fixture.lock
: > "$T/log"
log() { echo "$(date -u +%FT%TZ) $*" >> "$T/log"; }
# shellcheck source=rig-hold.sh
. "$here/rig-hold.sh"
fails=0
check() { if [ "$2" = 0 ]; then echo "RIGHOLD $1 -> PASS"; else echo "RIGHOLD $1 -> FAIL"; fails=$((fails + 1)); fi; }
lines() { command grep -c "$1" "$T/log"; }

# T1: back-to-back lock taker, 1 s per run.
( while [ ! -e "$T/stop" ]; do flock "$RIG_LOCK" sleep 1; done ) &
starver=$!
sleep 0.5
hits=0
for _ in $(seq 20); do flock -n "$RIG_LOCK" true && hits=$((hits + 1)); sleep 0.5; done
t0=$SECONDS
HOLD_POLL_S=1 rig_hold T1 $((SECONDS + 30))
r=$?
took=$((SECONDS - t0))
touch "$T/stop"
echo "RIGHOLD T1 reading: the old flock -n poll found the lock free on $hits of 20 probes over 10 s"
check "T1 hold gets in against a back-to-back taker rc=$r took=${took}s (want rc 0 within 10 s)" \
  $([ $r = 0 ] && [ $took -le 10 ] && echo 0 || echo 1)
rig_release T1
wait "$starver"; rm -f "$T/stop"

# T2: a compute app that clears after 3 s under the hold.
: > "$T/log"; touch "$T/app"; ( sleep 3; rm -f "$T/app" ) &
HOLD_POLL_S=1 HOLD_IDLE_S=10 rig_hold T2 $((SECONDS + 30))
r=$?
check "T2 app clears under the hold rc=$r held=$(lines 'held after') idle=$(lines 'idle under the hold') busy=$(lines 'busy under the hold') (want 0 1 1 0)" \
  $([ $r = 0 ] && [ "$(lines 'held after')" = 1 ] && [ "$(lines 'idle under the hold')" = 1 ] \
    && [ "$(lines 'busy under the hold')" = 0 ] && echo 0 || echo 1)

# T4: the hold is real, and a child with 8>&- does not inherit fd 8.
flock -n "$RIG_LOCK" true
other=$?
bash -c '[ -e /proc/$$/fd/8 ]' 8>&-
child=$?
bash -c '[ -e /proc/$$/fd/8 ]'
red=$?
check "T4 another flock -n while held rc=$other (want 1), child with 8>&- sees fd 8 rc=$child (want 1), without it rc=$red (want 0)" \
  $([ $other = 1 ] && [ $child = 1 ] && [ $red = 0 ] && echo 0 || echo 1)

# T5: release frees the lock.
rig_release T2
flock -n "$RIG_LOCK" true
r=$?
check "T5 flock -n after rig_release rc=$r (want 0)" $r

# T3: a compute app that never clears; 2 s under the hold, 1 s between tries, 8 s deadline.
: > "$T/log"; touch "$T/app"
( sleep 0.5; for _ in $(seq 35); do flock -n "$RIG_LOCK" true && { echo 0 > "$T/between"; exit; }; sleep 0.2; done
  echo 1 > "$T/between" ) &
between_job=$!
HOLD_POLL_S=1 HOLD_IDLE_S=2 HOLD_RETRY_S=1 rig_hold T3 $((SECONDS + 8))
r=$?
wait "$between_job" 2>/dev/null
flock -n "$RIG_LOCK" true
after=$?
busy=$(lines 'busy under the hold')
check "T3 app never clears rc=$r busy_lines=$busy free_between_tries=$(cat "$T/between" 2>/dev/null) free_after=$after (want 3 >=2 0 0)" \
  $([ $r = 3 ] && [ "$busy" -ge 2 ] && [ "$(cat "$T/between" 2>/dev/null)" = 0 ] && [ $after = 0 ] && echo 0 || echo 1)
rm -f "$T/app"

if [ $fails = 0 ]; then echo "RIGHOLD all checks -> PASS"; else echo "RIGHOLD $fails check(s) -> FAIL"; fi
[ $fails = 0 ]
