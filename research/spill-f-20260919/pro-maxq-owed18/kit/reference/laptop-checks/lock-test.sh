#!/usr/bin/env bash
# CPU-only tests of the kit's patches: runner-held lock FD in tier-battery, card-restricted
# telemetry, the external-lock handoff, and kit-procs reap/thermal-watch. Fake nvidia-smi on PATH.
set -u
T=/home/avifenesh/.cache/memra-maxq/handoff-direct/test
K=/home/avifenesh/.cache/memra-maxq/handoff-direct/kit
L=$T/locks/memra-gpu.lock
export PATH=$T/bin:$PATH FAKE_SMI_LOG=$T/smi.log MEMRA_TIER_BATTERY_LOCK_DIR=$T/locks
: > "$FAKE_SMI_LOG"
cd "$K" || exit 1
pass=0; fail=0
ok() { echo "PASS $*"; pass=$((pass + 1)); }
bad() { echo "FAIL $*"; fail=$((fail + 1)); }

# 1. Runner holds the lock on fd 9; tier-battery re-asserts it, the lock survives the cell.
exec 9>> "$L"
flock -n 9 || { echo "cannot take test lock"; exit 1; }
MAXQ_LOCK_FD=9 MAXQ_SMI_CARD=3 python3 tools/tier-battery.py --private-lock-dir-for-tests --rig pro-single \
  --timeout 30 --out "$T/run1" --execute bash -c 'flock -n '"$L"' true && echo OTHER-GOT-LOCK || echo other-refused; sleep 1' > "$T/run1.out" 2>&1
rc=$?
[ "$rc" = 0 ] && ok "1a collector ran under the runner lock" || bad "1a rc=$rc"
grep -q other-refused "$T/run1/command.log" && ok "1b an independent open was refused during the cell" || bad "1b"
flock -n "$L" true && bad "1c lock dropped after the cell" || ok "1c lock still held after the cell"
grep -q -- '-i 3 --query-gpu=' "$FAKE_SMI_LOG" && ok "1d telemetry sampler restricted to -i 3" || bad "1d"
grep -q -- '-i 3 --query-compute-apps' "$FAKE_SMI_LOG" && ok "1e compute-app snapshot restricted to -i 3" || bad "1e"
grep -q '"device": "3"' "$T/run1/command.capture.json" && ok "1f power limits recorded for card 3" || bad "1f"

# 2. External lock: the child gets a dup of the runner's description and can re-assert it.
MAXQ_LOCK_FD=9 MAXQ_SMI_CARD=3 python3 tools/tier-battery.py --private-lock-dir-for-tests --rig pro-single \
  --timeout 30 --external-lock --out "$T/run2" --execute python3 -c \
  'import fcntl, sys; fcntl.flock(int(sys.argv[1]), fcntl.LOCK_EX | fcntl.LOCK_NB); print("child-owns-description")' \
  @COLLECTOR_LOCK_FD@ > "$T/run2.out" 2>&1
grep -q child-owns-description "$T/run2/command.log" && ok "2a external-lock child owns the runner's description" || bad "2a"
exec 9>&-
flock -n "$L" true && ok "2b lock released when the runner closes fd 9" || bad "2b"

# 3. A runner FD that does not hold the lock while another description does: refused.
exec 8>> "$L"; flock -n 8
exec 7>> "$L"
MAXQ_LOCK_FD=7 MAXQ_SMI_CARD=3 python3 tools/tier-battery.py --private-lock-dir-for-tests --rig pro-single \
  --timeout 30 --out "$T/run3" --execute true > "$T/run3.out" 2>&1
rc=$?
[ "$rc" != 0 ] && grep -q "REFUSED: \[Errno 11\]" "$T/run3.out" && ok "3 refused while another description holds the lock (rc=$rc)" || bad "3 rc=$rc"
exec 7>&- 8>&-

# 4. MAXQ_LOCK_FD on a different inode: refused.
exec 6>> "$T/locks/other.lock"
MAXQ_LOCK_FD=6 python3 tools/tier-battery.py --private-lock-dir-for-tests --rig pro-single \
  --timeout 30 --out "$T/run4" --execute true > "$T/run4.out" 2>&1
grep -q "not the canonical lock inode" "$T/run4.out" && ok "4 wrong inode refused" || bad "4"
exec 6>&-

# 5. GpuSampler takes the card from MAXQ_SMI_CARD.
MAXQ_SMI_CARD=2 python3 -c 'import importlib.util as u; s = u.spec_from_file_location("g", "research/spill-f-20260919/m1-gpu-sampler.py"); m = u.module_from_spec(s); s.loader.exec_module(m); print(m.GpuSampler("/dev/null").device)' > "$T/run5.out" 2>&1
[ "$(cat "$T/run5.out")" = 2 ] && ok "5 GpuSampler device from MAXQ_SMI_CARD" || bad "5 $(cat "$T/run5.out")"

# 6. Reap: a tagged process in its own session dies; an untagged one survives.
MAXQ_RUN_ID=test-reap setsid sleep 300 &
TAGGED=$!
sleep 300 &
FREE=$!
sleep 0.5
python3 kit-procs.py reap --run-id test-reap --spare "$$" > "$T/run6.out" 2>&1
sleep 0.5
if ! kill -0 "$TAGGED" 2>/dev/null && kill -0 "$FREE" 2>/dev/null; then ok "6 reap kills only tagged processes"; else bad "6"; fi
kill "$FREE" 2>/dev/null; wait 2>/dev/null

# 7. Thermal watch: two hot rows do not stop, three in a row do; the spared PID stays.
CSV=$T/therm.csv
printf 'timestamp, temperature.gpu, fan.speed [%%]\n' > "$CSV"
MAXQ_RUN_ID=test-therm setsid sleep 300 &
VICTIM=$!
python3 kit-procs.py thermal-watch --csv "$CSV" --run-id test-therm --spare "$$" --out "$T/THERMAL-STOP.txt" > "$T/run7.out" 2>&1 &
W=$!
printf 't1, 93, 96 %%\nt2, 93, 96 %%\nt3, 91, 99 %%\nt4, 95, 94 %%\n' >> "$CSV"
sleep 2.5
[ ! -e "$T/THERMAL-STOP.txt" ] && ok "7a no stop below the rule (hot without full fan, or under 3 in a row)" || bad "7a"
printf 't5, 92, 95 %%\nt6, 94, 97 %%\nt7, 93, 100 %%\n' >> "$CSV"
sleep 3
[ -e "$T/THERMAL-STOP.txt" ] && ! kill -0 "$VICTIM" 2>/dev/null && ok "7b stop after 3 hot samples with fan >= 95, run reaped" || bad "7b"
kill "$W" "$VICTIM" 2>/dev/null; wait 2>/dev/null

echo "tests: $pass passed, $fail failed"
