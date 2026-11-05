#!/usr/bin/env bash
# Failure-path rehearsals of run.sh (stub servers, fake nvidia-smi, no GPU):
#   A no CARD -> REFUSED; B lock held elsewhere -> REFUSED, no out dir;
#   C SIGTERM during handoff-1g -> prompt stop, reaped, no-verdict VERDICT, scratch gone, lock free;
#   D 8 GiB projection over the time left -> handoff-8g.SKIPPED, no-verdict naming the skip.
set -u
REH=/home/avifenesh/.cache/memra-maxq/handoff-direct/rehearsal
R=$REH/hd
export PATH=$REH/bin:$PATH FAKE_SMI_LOG=$REH/smi-fail.log
pass=0; fail=0
ok() { echo "PASS $*"; pass=$((pass + 1)); }
bad() { echo "FAIL $*"; fail=$((fail + 1)); }
outs() { find "$R" -maxdepth 1 -name 'out-*' | wc -l; }

DL=$(TZ=<local-tz> date -d '+40 min' '+%Y-%m-%d %H:%M')
bash "$R/run.sh" > "$REH/failA.log" 2>&1
grep -q 'REFUSED: CARD=<nvidia-smi index> is required' "$REH/failA.log" && ok "A no CARD refused" || bad "A $(tail -1 "$REH/failA.log")"

n0=$(outs)
flock /tmp/memra-gpu.lock -c "CARD=3 DEADLINE='$DL' bash $R/run.sh" > "$REH/failB.log" 2>&1
grep -q 'REFUSED: /tmp/memra-gpu.lock is held by another job' "$REH/failB.log" && [ "$(outs)" = "$n0" ] \
  && ok "B lock held elsewhere refused, no out dir" || bad "B $(tail -1 "$REH/failB.log")"

CARD=3 DEADLINE="$DL" bash "$R/run.sh" > "$REH/failC.log" 2>&1 &
PID=$!
waitfor --grep 'handoff-1g start' "$REH/failC.log" --timeout 3m --quiet
waitfor --file "$R/scratch/b2/handoff.bin" --timeout 2m --every 1s --quiet
t0=$(date +%s)
kill -TERM "$PID"
wait "$PID"
rc=$?
dt=$(( $(date +%s) - t0 ))
OUTC=$(grep '^OUT ' "$REH/failC.log" | cut -d' ' -f2)
RUNC=$(basename "$OUTC" | sed 's/^out-//')
[ "$rc" = 130 ] && [ "$dt" -le 30 ] && ok "C SIGTERM stopped the run in ${dt}s rc=$rc" || bad "C rc=$rc dt=${dt}s"
grep -q 'default=none' "$OUTC/VERDICT.txt" && ok "C VERDICT no-verdict written" || bad "C verdict: $(tail -1 "$OUTC/VERDICT.txt")"
[ ! -e "$R/scratch" ] && ok "C scratch removed" || bad "C scratch left: $(ls -a "$R/scratch" "$R/scratch/b2" 2>&1 | tr '\n' ' ')"
flock -n /tmp/memra-gpu.lock true && ok "C lock free after stop" || bad "C lock still held"
left=$(python3 "$R/kit-procs.py" reap --run-id "$RUNC")
echo "$left" | grep -q 'SIGTERM 0, SIGKILL 0' && ok "C no tagged process left ($left)" || bad "C leftovers: $left"

sed -e 's/^EIGHT_OVER_ONE=3.0 /EIGHT_OVER_ONE=40.0 /' "$R/run.sh" > "$R/run-skip.sh"
DL2=$(TZ=<local-tz> date -d '+20 min' '+%Y-%m-%d %H:%M')
CARD=3 DEADLINE="$DL2" bash "$R/run-skip.sh" > "$REH/failD.log" 2>&1
OUTD=$(grep '^OUT ' "$REH/failD.log" | cut -d' ' -f2)
[ -f "$OUTD/handoff-8g.SKIPPED" ] && [ ! -d "$OUTD/handoff-8g" ] && ok "D 8 GiB cell not started: $(cat "$OUTD/handoff-8g.SKIPPED")" || bad "D"
# PREREG rule 3/4: with the 8 GiB cell absent, a counted 1 GiB winner gives default=none, a flat or loser gives default=buffered.
V=$(tail -1 "$OUTD/VERDICT.txt")
if echo "$V" | grep -q 'handoff-1g=winner'; then want='default=none'; else want='default=buffered'; fi
echo "$V" | grep -q "handoff-8g=absent $want (handoff-8g: handoff-8g not started" && ok "D verdict follows the rule ($want) and names the skip" || bad "D verdict: $V"
echo "fail-path rehearsals: $pass passed, $fail failed"
