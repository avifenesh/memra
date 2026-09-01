#!/usr/bin/env bash
# test_local_ci_lock.sh — teeth for local-ci's whole-run GPU lock (lane/localci-lock-20260821).
#
# WHY THIS EXISTS: on 2026-08-21 two lanes independently hit the same pair of scheduling
# bugs — (1) local-ci ran prime-gate and every other correctness-stage GPU step with NO
# lock, so a foreign 10GB co-resident OOMed the battery (fail-open); (2) wrapping the whole
# battery in `flock /tmp/memra-5090.lock tools/local-ci.sh` SELF-DEADLOCKED, because
# spec-on-cache-hit-gate's internal per-boot `flock -w 300` contends the same file and
# MEMRA_CI_LOCK_HELD=1 never covered it. The fix is one fd-held lock at the top of
# local-ci's GPU section plus a distinct inner lock seam for the hit-gate. This harness
# proves the mechanics — CPU-only, private lock files, no GPU, no model, no real rig lock.
#
# What it asserts:
#   A. two concurrent local-ci invocations SERIALIZE: the second waits loudly, both green.
#   B. killing local-ci mid-hold RELEASES the lock (fd-based flock, kernel cleanup —
#      no stale-lockfile state to scrub).
#   C. the caller-held pattern (MEMRA_CI_LOCK_HELD=1) does not re-acquire AND still
#      redirects MEMRA_GPU_LOCK off the held file — the exact footgun that self-deadlocked
#      the hit-gate when an operator remembered the outer lock but not the private file.
#   D. an explicit caller MEMRA_GPU_LOCK is respected, never clobbered.
#   E. the inner gate (spec-on-cache-hit-gate) invoked STANDALONE still takes its own lock:
#      with the lock held elsewhere it BLOCKS at boot() instead of running lock-less.
#
# Exercises the REAL scripts via local-ci's MEMRA_CI_LOCK_SMOKE door (real acquisition
# path, no build, no GPU). Runs inside the battery (fatal) and standalone.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

# MEMRA_LOCKTEST_CI: teeth seam — point the harness at a deliberately broken local-ci
# copy (flock removed / exports dropped) and every arm here must go RED. A lock harness
# only ever observed passing proves nothing (the gate-teeth law).
CI=${MEMRA_LOCKTEST_CI:-tools/local-ci.sh}
HITGATE=tools/spec-on-cache-hit-gate.sh
WORK=$(mktemp -d /tmp/memra-localci-locktest.XXXXXX) || exit 1
PIDS=()
# Kill a spawned script AND its live children (the blocked flock / the holding sleep).
# Children first, while the parent still names them via --ppid; a SIGKILLed parent's
# orphans re-parent to init and become unfindable. SIGKILL on purpose: it models the
# worst-case death (no traps run) — the lock must survive exactly that.
kill_tree() {
    local pid=$1 kid
    # Process depth differs across shells and container launchers: the lock-holding
    # sleep may be a grandchild rather than a direct child. Walk descendants first so
    # SIGKILL cannot orphan a deeper process that inherited the lock fd.
    while read -r kid; do
        [ -n "$kid" ] && kill_tree "$kid"
    done < <(ps -o pid= --ppid "$pid" 2>/dev/null)
    kill -9 "$pid" 2>/dev/null
}
cleanup() {
    local p
    for p in "${PIDS[@]:-}"; do
        [ -n "$p" ] && kill_tree "$p"
    done
    rm -rf "$WORK"
}
trap cleanup EXIT
FAILS=0
ok()   { echo "  ok: $1"; }
fail() { echo "  FAIL: $1"; FAILS=$((FAILS + 1)); }

# Spawn a lock-smoke local-ci with a SCRUBBED env: a battery run wires this harness in
# before it acquires, but a caller (or a parent battery) may already export
# MEMRA_CI_LOCK_HELD/MEMRA_GPU_LOCK — inherited, those would make every child skip the
# very acquisition under test.
smoke() { # $1 hold-seconds, then extra VAR=val pairs; stdin/stdout as caller wires them
    local hold=$1; shift
    env -u MEMRA_CI_LOCK_HELD -u MEMRA_GPU_LOCK \
        MEMRA_CI_LOCK="$WORK/ci.lock" MEMRA_CI_LOCK_WAIT=30 \
        MEMRA_CI_LOCK_SMOKE=1 MEMRA_CI_LOCK_SMOKE_HOLD="$hold" \
        "$@" bash "$CI"
}

echo "== test A: two concurrent local-ci invocations serialize (second waits, both green) =="
start_a=$(date +%s)
smoke 6 > "$WORK/a.log" 2>&1 &
A_PID=$!
# Wait until A actually holds (its acquired line is printed after flock succeeds).
for _ in $(seq 1 50); do grep -q "GPU lock acquired" "$WORK/a.log" 2>/dev/null && break; sleep 0.2; done
grep -q "GPU lock acquired" "$WORK/a.log" || fail "A never acquired the private lock"
start_b=$(date +%s)
smoke 0 > "$WORK/b.log" 2>&1
b_rc=$?
end_b=$(date +%s)
wait "$A_PID"; a_rc=$?
[ "$a_rc" -eq 0 ] && ok "first run green (rc=0)" || fail "first run rc=$a_rc"
[ "$b_rc" -eq 0 ] && ok "second run green (rc=0)" || fail "second run rc=$b_rc"
if grep -q "HELD by another run — WAITING" "$WORK/b.log"; then
    ok "second run announced the wait loudly"
else
    fail "second run did not print the waiting message (silent wait or lock-less run)"
    sed 's/^/    b| /' "$WORK/b.log"
fi
# A holds ~6s from its start; B began ~immediately after A's acquire, so B must have
# waited several seconds rather than running concurrently. 3s margin absorbs slow forks.
if [ $((end_b - start_b)) -ge 3 ]; then
    ok "second run actually waited ($((end_b - start_b))s; first held 6s from t+$((start_b - start_a))s)"
else
    fail "second run finished in $((end_b - start_b))s while the first held the lock — no serialization"
fi

echo "== test B: killing local-ci mid-run releases the lock =="
smoke 120 > "$WORK/kill.log" 2>&1 &
K_PID=$!
PIDS+=("$K_PID")
busy=0
for _ in $(seq 1 50); do
    if ! flock -n "$WORK/ci.lock" -c true 2>/dev/null; then busy=1; break; fi
    sleep 0.2
done
[ "$busy" -eq 1 ] && ok "lock observed held mid-run" || fail "run never took the lock"
{ kill_tree "$K_PID"; wait "$K_PID"; } 2>/dev/null
released=0
for _ in $(seq 1 25); do
    if flock -n "$WORK/ci.lock" -c true 2>/dev/null; then released=1; break; fi
    sleep 0.2
done
[ "$released" -eq 1 ] && ok "SIGKILL released the lock (fd-based, kernel cleanup)" \
                      || fail "lock still held after SIGKILL — stale-lock hazard"

echo "== test C: caller-held pattern skips re-acquire AND redirects the inner seam =="
flock -x "$WORK/ci.lock" -c 'sleep 20' &
HOLDER=$!
sleep 0.3
start_c=$(date +%s)
env -u MEMRA_GPU_LOCK MEMRA_CI_LOCK_HELD=1 \
    MEMRA_CI_LOCK="$WORK/ci.lock" MEMRA_CI_LOCK_WAIT=30 \
    MEMRA_CI_LOCK_SMOKE=1 MEMRA_CI_LOCK_SMOKE_HOLD=0 \
    bash "$CI" > "$WORK/c.log" 2>&1
c_rc=$?
end_c=$(date +%s)
kill_tree "$HOLDER"; wait "$HOLDER" 2>/dev/null
[ "$c_rc" -eq 0 ] && [ $((end_c - start_c)) -lt 10 ] \
    && ok "MEMRA_CI_LOCK_HELD=1 ran immediately against a held lock (no second take, no deadlock)" \
    || fail "caller-held run rc=$c_rc elapsed=$((end_c - start_c))s — it re-contended the held lock"
grep -q "MEMRA_GPU_LOCK=$WORK/ci.lock.inner" "$WORK/c.log" \
    && ok "inner seam redirected off the held file (hit-gate cannot self-deadlock)" \
    || { fail "MEMRA_GPU_LOCK not redirected under MEMRA_CI_LOCK_HELD=1"; sed 's/^/    c| /' "$WORK/c.log"; }

echo "== test D: an explicit caller MEMRA_GPU_LOCK is respected =="
smoke 0 MEMRA_GPU_LOCK="$WORK/private.lock" > "$WORK/d.log" 2>&1
grep -q "MEMRA_GPU_LOCK=$WORK/private.lock\$" "$WORK/d.log" \
    && ok "explicit private lock survived" \
    || { fail "explicit MEMRA_GPU_LOCK was clobbered"; sed 's/^/    d| /' "$WORK/d.log"; }
# and the footgun value — MEMRA_GPU_LOCK pointed AT the whole-run lock — is redirected:
smoke 0 MEMRA_GPU_LOCK="$WORK/ci.lock" > "$WORK/d2.log" 2>&1
grep -q "MEMRA_GPU_LOCK=$WORK/ci.lock.inner" "$WORK/d2.log" \
    && ok "MEMRA_GPU_LOCK==whole-run lock is redirected (the self-deadlock value)" \
    || { fail "MEMRA_GPU_LOCK equal to the held lock was kept — self-deadlock preserved"; sed 's/^/    d2| /' "$WORK/d2.log"; }

echo "== test E: spec-on-cache-hit-gate standalone still takes its own lock =="
# Hold the gate's lock; the gate must BLOCK at boot()'s flock rather than run lock-less.
# Hermetic: fake model, /bin/false as the server (never reached while blocked), a
# quiet port, and a no-op pkill on PATH so the gate's stop() trap cannot touch real
# memra-server processes even on the failure path.
mkdir -p "$WORK/fakebin" "$WORK/ev"
printf '#!/bin/sh\nexit 0\n' > "$WORK/fakebin/pkill" && chmod +x "$WORK/fakebin/pkill"
: > "$WORK/fake-model.gguf"
flock -x "$WORK/gate.lock" -c 'sleep 60' &
GHOLDER=$!
sleep 0.3
env -u MEMRA_CI_LOCK_HELD PATH="$WORK/fakebin:$PATH" \
    MEMRA_GPU_LOCK="$WORK/gate.lock" MEMRA_GATE_PORT=18741 \
    bash "$HITGATE" qwen "$WORK/fake-model.gguf" /bin/false "$WORK/ev" \
    > "$WORK/hitgate.log" 2>&1 &
G_PID=$!
PIDS+=("$G_PID")
sleep 4
if kill -0 "$G_PID" 2>/dev/null \
        && pgrep -f "flock.*$WORK/gate.lock" >/dev/null \
        && ! grep -qE "server died during boot|server never became ready" "$WORK/hitgate.log"; then
    ok "standalone hit-gate blocked on its own lock (no lock-less boot)"
else
    fail "standalone hit-gate proceeded past a held MEMRA_GPU_LOCK (lock-less GPU boot)"
    sed 's/^/    e| /' "$WORK/hitgate.log" | tail -20
fi
{ kill_tree "$G_PID"; wait "$G_PID"; } 2>/dev/null
kill_tree "$GHOLDER"; wait "$GHOLDER" 2>/dev/null

if [ "$FAILS" -eq 0 ]; then
    echo "test_local_ci_lock: ALL GREEN"
    exit 0
fi
echo "test_local_ci_lock: $FAILS FAIL"
exit 1
