#!/usr/bin/env bash
# tools/test_gate_template_integrity.sh — teeth for round 3 of GATE-INTEGRITY-20260819.
#
# WHAT THIS FIXTURE IS FOR. Rounds 1 and 2 fixed gates that were written before the rule
# existed. This round fixes the TEMPLATE those gates get generated from, which is where the rule
# stops needing to be re-learned — and a template fix without a generated-artifact test is just a
# diff. So every runtime arm below runs a gate that the generator actually EMITTED, against a
# stub server on a real loopback port, and forces the failure.
#
# CPU ONLY. No GPU, no model, no CUDA toolkit, no network: throwaway repos under mktemp, a stub
# `memra-server` that is a python http.server, a stub `nvidia-smi`, a stub `cargo` that prints
# canned libtest output, and one real loopback listener.
#
# BOTH DIRECTIONS. Point MEMRA_GATE_SRC_DIR at another checkout (e.g. a worktree of origin/main)
# and the fixture renders its gates from THAT tree's generator and templates. Read the pre-fix
# score with the caveat printed at the end: origin/main's generator REJECTS a spec carrying
# batch.canary_expect_regex, so the fixture falls back to a v1 spec to keep the artifact arms
# decisive about the ARTIFACT rather than all failing on one schema error.
#
# VERDICTS GO TO A FILE AND THE COUNT IS AN EQUALITY. Round 1's first fixture printed
# "1 passed / 0 failed" with an arm visibly FAILing, because subshell counters are discarded.
# Every arm appends to $VERDICTS, the file is counted, and a total that misses the declared
# constant exits 3 as a BROKEN FIXTURE — an arm that quietly stops running must red the run, not
# shrink it.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GATE_SRC="${MEMRA_GATE_SRC_DIR:-$ROOT}"
EXPECT_ASSERTIONS=51

WORK=$(mktemp -d "${TMPDIR:-/tmp}/gate-template-fixture-XXXXXX")
VERDICTS="$WORK/verdicts.txt"
: > "$VERDICTS"
cleanup() {
    # Any stub server or listener we started lives in this process group; kill by pidfile.
    for pf in "$WORK"/*.pid; do
        [ -f "$pf" ] && kill "$(cat "$pf")" 2>/dev/null
    done
    rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

pass() { printf 'PASS\t%s\n' "$1" >> "$VERDICTS"; echo "ok   $1"; }
fail() { printf 'FAIL\t%s\n' "$1" >> "$VERDICTS"; echo "FAIL $1"; [ -n "${2:-}" ] && echo "       $2"; }

# assert <name> <condition-rc> [detail]
assert() { if [ "$2" -eq 0 ]; then pass "$1"; else fail "$1" "${3:-}"; fi; }
# assert_grep <name> <pattern> <file>
assert_grep() {
    if grep -qE "$2" "$3" 2>/dev/null; then pass "$1"
    else fail "$1" "pattern not found: $2 (in $3)"; fi
}
assert_not_grep() {
    if grep -qE "$2" "$3" 2>/dev/null; then fail "$1" "pattern present but must not be: $2"
    else pass "$1"; fi
}
assert_rc() { # <name> <expected> <actual>
    if [ "$2" = "$3" ]; then pass "$1"; else fail "$1" "expected rc $2, got $3"; fi
}

echo "=== gate-template integrity fixture — source tree: $GATE_SRC ==="

# ---------------------------------------------------------------------------
# The real tree's port census, computed here rather than taken from the generator, so the
# collision arm is decisive even against a generator that has no --list-ports.
# ---------------------------------------------------------------------------
REAL_PORTS="$WORK/real-ports.txt"
grep -hoE '^[A-Z_]*PORT=.*[0-9]{4,5}' "$ROOT"/tools/*.sh 2>/dev/null \
    | grep -oE '[0-9]{4,5}' | sort -u > "$REAL_PORTS"
REAL_PORT_COUNT=$(grep -c . "$REAL_PORTS")
if [ "$REAL_PORT_COUNT" -lt 10 ]; then
    echo "BROKEN FIXTURE: the real-tree port scan found $REAL_PORT_COUNT ports (floor 10)." >&2
    echo "  Every collision arm below would pass vacuously against an empty list." >&2
    exit 3
fi
echo "real-tree fixed ports: $REAL_PORT_COUNT distinct"

# ---------------------------------------------------------------------------
# Sandbox: a throwaway repo the generator can write into, carrying enough stub gates that its
# own port census clears its non-vacuity floor without pointing at the real tree.
# ---------------------------------------------------------------------------
SANDBOX="$WORK/sandbox"
mkdir -p "$SANDBOX/tools/arch-gate-templates" "$SANDBOX/target/release"
: > "$SANDBOX/Cargo.toml"
cp "$GATE_SRC/tools/generate-arch-gates.py" "$SANDBOX/tools/"
cp "$GATE_SRC"/tools/arch-gate-templates/* "$SANDBOX/tools/arch-gate-templates/"
[ -f "$GATE_SRC/tools/port-guard.sh" ] && cp "$GATE_SRC/tools/port-guard.sh" "$SANDBOX/tools/"
for n in $(seq -w 1 12); do
    printf 'PORT=190%s\n' "$n" > "$SANDBOX/tools/portstub-$n.sh"
done

MODEL="$WORK/model.gguf"
DRAFT="$WORK/draft.gguf"
: > "$MODEL"
: > "$DRAFT"

write_spec() { # $1 out  $2 port  $3 with-canary-expect(0/1)  $4 draft_path
    local expect_line=""
    [ "$3" = 1 ] && expect_line='"canary_expect_regex": "FAIL: no batched-walk evidence",'
    cat > "$1" <<JSON
{
  "id": "tf",
  "artifact_env": "FIXTURE_ARTIFACT",
  "chunk": {"label": "tf-chunk", "prompts": ["research/p.txt"], "chunks": [512, 64],
            "steps": 4, "seam": "FIXTURE_CHUNK_SEAM"},
  "tick": {"label": "tf-tick", "prompts": ["research/p.txt"], "budgets": [0, 64],
           "splits": [64], "steps": 4, "seam": "FIXTURE_TICK_SEAM"},
  "batch": {
    "model_alias": "tfmodel",
    "draft_path": $4,
    "draft_env": "FIXTURE_DRAFT",
    "canary_env": {"FIXTURE_BATCH": "0"},
    $expect_line
    "required_gpus": 2, "pp_stages": 2, "pp_devices": [0, 1],
    "concurrency": [2],
    "port": $2,
    "receipt_dir": "receipts",
    "server_env": {"FIXTURE_SERVE": "1"},
    "request": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
                "temperature": 0.0},
    "liveness": {"cap_regex": "tfmodel: decode chunk cap [0-9]+", "cap_min": 2,
                 "walk_regex": "\\\\[tfmodel-batch\\\\] first B>1"}
  },
  "mapping": [{"path_regex": "^crates/", "kernel_scope": "none", "base_probes": ["x"],
               "base_spec_probes": [], "gate_families": ["chunk", "tick", "batch"]}]
}
JSON
}

# v2 first (the fixed schema). If the generator refuses the discriminator it is a pre-fix copy,
# so fall back to v1 with the OLD documented port — the artifact arms then measure the artifact.
GEN_MODE=v2
write_spec "$WORK/spec.json" 18300 1 "\"$DRAFT\""
GENLOG="$WORK/generate.log"
if ! python3 "$SANDBOX/tools/generate-arch-gates.py" "TF Arch" "$MODEL" \
        --spec "$WORK/spec.json" --out-dir "$SANDBOX/tools/generated-arch-gates/tf-arch" \
        > "$GENLOG" 2>&1; then
    GEN_MODE=v1
    write_spec "$WORK/spec.json" 8094 0 "\"$DRAFT\""
    python3 "$SANDBOX/tools/generate-arch-gates.py" "TF Arch" "$MODEL" \
        --spec "$WORK/spec.json" --out-dir "$SANDBOX/tools/generated-arch-gates/tf-arch" \
        > "$GENLOG" 2>&1
fi
GATE="$SANDBOX/tools/generated-arch-gates/tf-arch/tf-arch-b2-geometry-gate.sh"
echo "generator schema accepted: $GEN_MODE"

assert "generator emitted a b2-geometry gate" "$([ -x "$GATE" ] && echo 0 || echo 1)" \
    "$(tail -3 "$GENLOG")"
if [ ! -x "$GATE" ]; then
    echo "BROKEN FIXTURE: no gate to test." >&2
    cat "$GENLOG" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# STATIC arms on the emitted text. Comments are stripped first: the fixed template DOCUMENTS
# 8094 and the old shapes verbatim, and matching the documentation of a bug as the bug is its
# own blind assertion (round 2 hit this exact trap).
# ---------------------------------------------------------------------------
CODE="$WORK/gate-code.sh"
grep -vE '^\s*#' "$GATE" > "$CODE"

GATE_PORT=$(grep -oE '^DEFAULT_PORT=[0-9]+' "$CODE" | head -1 | cut -d= -f2)
if [ -n "$GATE_PORT" ] && [ "$GATE_PORT" -ge 18300 ] && [ "$GATE_PORT" -le 18399 ]; then
    pass "generated port $GATE_PORT is inside the reserved 18300-18399 band"
else
    fail "generated port is inside the reserved 18300-18399 band" "got '${GATE_PORT:-unset}'"
fi
if [ -n "$GATE_PORT" ] && grep -qx "$GATE_PORT" "$REAL_PORTS"; then
    fail "generated port does not collide with a hand-written gate" \
        "$GATE_PORT is bound by: $(grep -lE "PORT=.*$GATE_PORT" "$ROOT"/tools/*.sh | xargs -n1 basename | tr '\n' ' ')"
else
    pass "generated port does not collide with a hand-written gate ($REAL_PORT_COUNT scanned)"
fi
assert_grep "per-gate port override variable is derived, not shared" \
    '^PORT_ENV=MEMRA_[A-Z0-9_]+_B2GEO_PORT$' "$CODE"
assert_grep "tools/port-guard.sh is sourced" 'port-guard\.sh' "$CODE"
assert_grep "the gate refuses when port-guard.sh is missing" \
    'port-guard\.sh is missing' "$GATE"
assert_grep "ask\(\) refuses an empty completion (A-15)" 'empty_completion' "$CODE"
assert_grep "assertion count is declared (1 ref + 2 responses + 2 liveness)" \
    '^EXPECT_ASSERTS=5$' "$CODE"
# grep is line-oriented, so the multi-line shape has to be matched with a context window. The
# first draft of this arm used a pattern with `\n` in it and could never match anything, which
# made it pass against the pre-fix template too — an assertion that cannot fail, in the fixture
# for a defect class whose second member is exactly that.
if grep -A2 -E 'SKIP \(' "$CODE" | grep -qE '^\s*exit 0\s*$'; then
    fail "no SKIP path exits 0 within two lines of printing SKIP" \
        "$(grep -A2 -E 'SKIP \(' "$CODE" | grep -nE '^\s*exit 0\s*$' | head -2 | tr '\n' ' ')"
else
    pass "no SKIP path exits 0 within two lines of printing SKIP"
fi
assert_grep "scratch dir is trapped for cleanup (/tmp hygiene law)" \
    "trap .*rm -rf .\\\$TMP" "$CODE"

# guard BEFORE the bind, ownership AFTER it — by line number, not by eye.
G_LINE=$(grep -n 'memra_port_guard' "$CODE" | head -1 | cut -d: -f1)
B_LINE=$(grep -n 'MEMRA_ADDR=' "$CODE" | head -1 | cut -d: -f1)
O_LINE=$(grep -n 'memra_port_owned' "$CODE" | head -1 | cut -d: -f1)
if [ -n "$G_LINE" ] && [ -n "$B_LINE" ] && [ "$G_LINE" -lt "$B_LINE" ]; then
    pass "port guard runs BEFORE the server bind (line $G_LINE < $B_LINE)"
else
    fail "port guard runs BEFORE the server bind" "guard=${G_LINE:-none} bind=${B_LINE:-none}"
fi
if [ -n "$O_LINE" ] && [ -n "$B_LINE" ] && [ "$O_LINE" -gt "$B_LINE" ]; then
    pass "post-boot pid ownership is asserted (line $O_LINE > $B_LINE)"
else
    fail "post-boot pid ownership is asserted" "owned=${O_LINE:-none} bind=${B_LINE:-none}"
fi

# ---------------------------------------------------------------------------
# Stubs. The server is a real HTTP server so the gate's own curl/readyz/ownership path runs.
# ---------------------------------------------------------------------------
BINDIR="$WORK/bin"
mkdir -p "$BINDIR"
cat > "$BINDIR/nvidia-smi" <<'SH'
#!/usr/bin/env bash
[ "${FIXTURE_NVSMI_FAIL:-0}" = 1 ] && { echo "Unable to determine the device handle" >&2; exit 9; }
case "$*" in
  *index,memory.used*) printf '0, 100 MiB\n1, 100 MiB\n' ;;
  *) printf '0\n1\n' ;;
esac
SH
chmod +x "$BINDIR/nvidia-smi"

# A PATH with everything the gate needs EXCEPT nvidia-smi. `PATH=/nowhere` would test nothing
# useful: the gate would die at 127 on the first `date`, and "the gate failed" is not "the gate
# refused because it cannot see a GPU". The list is explicit so a missing entry shows up as a
# 127 rather than as a silently different code path.
NOSMI="$WORK/nosmi-bin"
mkdir -p "$NOSMI"
for tool in bash sh env dirname basename date mkdir mktemp flock curl python3 tee cat sed \
            seq sleep grep rm touch head cut tr sort wc ss lsof; do
    real=$(command -v "$tool" 2>/dev/null) && ln -sf "$real" "$NOSMI/$tool"
done
for needed in bash date mktemp flock python3 grep sed; do
    [ -x "$NOSMI/$needed" ] || { echo "BROKEN FIXTURE: no $needed on PATH" >&2; exit 3; }
done

cat > "$SANDBOX/target/release/memra-server" <<'PY'
#!/usr/bin/env python3
"""Stub memra-server: prints the liveness lines the gate greps, then serves the two routes.

FIXTURE_WALK=0            omit the [tfmodel-batch] first B>1 line (breaks the subject)
FIXTURE_NULL=1            answer with {"reasoning": null, "content": null} (the A-15 shape)
FIXTURE_DIVERGE=1         answer differently on every request (breaks byte identity)
FIXTURE_BATCH=0           the canary seam: same effect as FIXTURE_WALK=0
"""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

print("tfmodel: decode chunk cap 4", flush=True)
seam_off = os.environ.get("FIXTURE_BATCH", "1") == "0" \
    and os.environ.get("FIXTURE_IGNORE_SEAM") != "1"
walk = os.environ.get("FIXTURE_WALK", "1") == "1" and not seam_off
if walk:
    print("[tfmodel-batch] first B>1 ready=2", flush=True)

COUNT = [0]
LOCK = threading.Lock()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        if os.environ.get("FIXTURE_NULL") == "1":
            msg = {"reasoning": None, "content": None}
        elif os.environ.get("FIXTURE_DIVERGE") == "1":
            with LOCK:
                COUNT[0] += 1
                msg = {"reasoning": None, "content": "answer-%d" % COUNT[0]}
        else:
            msg = {"reasoning": None, "content": "steady"}
        body = json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}],
                           "usage": {"completion_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


addr = os.environ.get("MEMRA_ADDR", "127.0.0.1:0")
host, _, port = addr.rpartition(":")
ThreadingHTTPServer((host, int(port)), H).serve_forever()
PY
chmod +x "$SANDBOX/target/release/memra-server"

GATE_ENV=(
    "PATH=$BINDIR:$PATH"
    "FIXTURE_ARTIFACT=$MODEL"
    "MEMRA_GPU_LOCK=$WORK/gpu.lock"
    "MEMRA_GPU_LOCK_WAIT=5"
)
run_gate() { # extra env..., then --  then gate args
    local envs=() out
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift || true
    out="$WORK/run-$RANDOM.out"
    ( cd "$SANDBOX" && env "${GATE_ENV[@]}" "${envs[@]}" "$GATE" "$@" ) > "$out" 2>&1
    LAST_RC=$?
    LAST_OUT="$out"
}

# ---- skip contract ----
run_gate "FIXTURE_ARTIFACT=$WORK/absent.gguf" --
assert_rc "missing artifact is FATAL (exit 77), not a silent pass" 77 "$LAST_RC"
assert_grep "the refusal says a skip is not a pass" 'a skip is not a pass' "$LAST_OUT"

run_gate "FIXTURE_ARTIFACT=$WORK/absent.gguf" "MEMRA_ARCH_GATE_ALLOW_SKIP=1" --
assert_rc "an EXPLICITLY accounted skip is allowed (the developer escape hatch)" 0 "$LAST_RC"
assert_grep "an accounted skip says the run proves nothing" 'skip ACCOUNTED' "$LAST_OUT"

CENSUS="$WORK/skip-census.tsv"
run_gate "FIXTURE_ARTIFACT=$WORK/absent.gguf" "MEMRA_ARCH_GATE_ALLOW_SKIP=1" \
    "MEMRA_SKIP_CENSUS=$CENSUS" --
if [ "$(grep -c . "$CENSUS" 2>/dev/null || echo 0)" = 1 ]; then
    pass "the skip is recorded in MEMRA_SKIP_CENSUS (one row)"
else
    fail "the skip is recorded in MEMRA_SKIP_CENSUS (one row)" \
        "rows=$(grep -c . "$CENSUS" 2>/dev/null || echo 0)"
fi

run_gate "FIXTURE_DRAFT=$WORK/absent-draft.gguf" --
assert_rc "an env-NAMED drafter that is absent is a hard FAIL, not a plain boot" 1 "$LAST_RC"
assert_grep "the drafter refusal refuses to boot plain" 'Refusing to boot plain' "$LAST_OUT"

run_gate "PATH=$NOSMI" --
assert_rc "no nvidia-smi is a censused skip (77), not exit 0" 77 "$LAST_RC"

run_gate "FIXTURE_NVSMI_FAIL=1" --
assert_rc "an nvidia-smi that FAILS is a FAIL, not 'a box with 0 GPUs'" 1 "$LAST_RC"
assert_grep "the nvidia-smi refusal names the distinction" \
    'not a GPU-less box' "$LAST_OUT"

# ---- the control: a healthy run must PASS, or every arm above could be "always refuse" ----
run_gate --
assert_rc "CONTROL: a healthy stub server passes the gate" 0 "$LAST_RC"
assert_grep "CONTROL: the PASS verdict names all 5 assertions" \
    'VERDICT: PASS \(5 assertions' "$LAST_OUT"

# ---- port guard, against a real listener on the gate's own port ----
python3 - "$GATE_PORT" "$WORK/listener.pid" <<'PY' &
import socket, sys, os, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(1)
open(sys.argv[2], "w").write(str(os.getpid()))
time.sleep(120)
PY
LISTENER=$!
echo "$LISTENER" > "$WORK/listener.pid"
for _ in 1 2 3 4 5 6 7 8 9 10; do
    ss -tln 2>/dev/null | grep -q "[:.]$GATE_PORT " && break
    sleep 0.3
done
run_gate --
assert_rc "an occupied port makes the gate REFUSE (not measure a stranger)" 1 "$LAST_RC"
assert_grep "the refusal names the port" "port $GATE_PORT is already LISTENing" "$LAST_OUT"
assert_grep "the refusal names the per-gate override variable" \
    'MEMRA_[A-Z0-9_]+_B2GEO_PORT=<free port>' "$LAST_OUT"
assert_not_grep "the gate did not boot a server behind the refusal" \
    'lock released' "$LAST_OUT"
kill "$LISTENER" 2>/dev/null
wait "$LISTENER" 2>/dev/null
rm -f "$WORK/listener.pid"

# ---- THE RECEIPTED INCIDENT: a foreign responder that ANSWERS ----
# tools/accept-gate.sh records it verbatim: "the rig's idle llama-server happened to hold 8181,
# so /health answered INSTANTLY from a foreign process, our own server was never waited for...
# Had that foreign process instead answered 200 with a plausible body, the gate would have
# measured SOMEONE ELSE'S MODEL and pinned it."
#
# A raw socket (the arm above) is not that incident: the unguarded gate's own server merely fails
# to bind and dies, and the gate reds by accident. Here the squatter speaks the API. Without the
# guard the readyz probe is satisfied by the stranger on its FIRST iteration, the dead child is
# never noticed, and every byte-identity assertion is satisfied by the stranger's answers. The
# decisive evidence is therefore not the exit code — it is whether the gate ever printed a
# comparison verdict at all.
FOREIGN_LOG="$WORK/foreign.log"
FIXTURE_STUB_ADDR="127.0.0.1:$GATE_PORT" \
    MEMRA_ADDR="127.0.0.1:$GATE_PORT" \
    python3 "$SANDBOX/target/release/memra-server" > "$FOREIGN_LOG" 2>&1 &
FOREIGN=$!
echo "$FOREIGN" > "$WORK/foreign.pid"
for _ in 1 2 3 4 5 6 7 8 9 10; do
    curl -sf "http://127.0.0.1:$GATE_PORT/readyz" >/dev/null 2>&1 && break
    sleep 0.3
done
run_gate --
if [ "$LAST_RC" -ne 0 ] && grep -q 'already LISTENing' "$LAST_OUT"; then
    pass "a foreign responder that SPEAKS the API is refused before boot"
else
    fail "a foreign responder that SPEAKS the API is refused before boot" "rc=$LAST_RC"
fi
assert_not_grep "the gate never compares a STRANGER's answers (no '== ref' verdict)" \
    '== ref' "$LAST_OUT"
kill "$FOREIGN" 2>/dev/null
wait "$FOREIGN" 2>/dev/null
rm -f "$WORK/foreign.pid"

# ---- port guard blind: no ss and no lsof ----
if [ -f "$SANDBOX/tools/port-guard.sh" ]; then
    PG_OUT="$WORK/pg-blind.out"
    # Absolute $BASH: `PATH=/nonexistent bash ...` cannot find bash either, and 127 is not
    # the refusal under test.
    ( cd "$SANDBOX" && PATH=/nonexistent "$BASH" tools/port-guard.sh check tf "$GATE_PORT" X ) \
        > "$PG_OUT" 2>&1
    PG_RC=$?
    assert_rc "no ss and no lsof is rc 2, not 'the port is free'" 2 "$PG_RC"
    assert_grep "the blind refusal says it cannot observe" \
        'cannot observe listening sockets' "$PG_OUT"
else
    fail "no ss and no lsof is rc 2, not 'the port is free'" "tools/port-guard.sh absent in $GATE_SRC"
    fail "the blind refusal says it cannot observe" "tools/port-guard.sh absent in $GATE_SRC"
fi

# ---- the subject is broken: no B>1 walk evidence ----
run_gate "FIXTURE_WALK=0" --
assert_rc "a broken subject reds the gate" 1 "$LAST_RC"
assert_grep "and it reds for the RIGHT reason (the batched-walk arm)" \
    'FAIL: no batched-walk evidence' "$LAST_OUT"

# ---- A-15: a 200 with a null completion must not hold vacuously ----
run_gate "FIXTURE_NULL=1" --
if [ "$LAST_RC" -ne 0 ]; then
    pass "an all-null completion cannot produce a PASS (A-15)"
else
    fail "an all-null completion cannot produce a PASS (A-15)" \
        "the gate exited 0 against {\"reasoning\": null, \"content\": null}"
fi
assert_not_grep "and the byte-identity headline is not claimed against a null reference" \
    'VERDICT: PASS' "$LAST_OUT"

# ---- canary arms ----
run_gate -- --canary
assert_rc "canary: the seam bites and the canary passes" 0 "$LAST_RC"
assert_grep "canary: it names the DECLARED discriminator, not just 'nonzero exit'" \
    'CANARY OK \(the rollback seam broke the declared assertion' "$LAST_OUT"

# FIXTURE_IGNORE_SEAM keeps the batched-walk line even under the canary env, so the run goes
# red for a DIFFERENT reason than the seam guarantees. That is the A-10 shape: a canary that
# reads "nonzero exit" would certify the naked arm from this.
run_gate "FIXTURE_DIVERGE=1" "FIXTURE_IGNORE_SEAM=1" -- --canary
assert_rc "canary: a red for the WRONG reason is not a canary pass" 1 "$LAST_RC"
assert_grep "canary: it says which check went red instead" \
    'the run went red for the WRONG reason' "$LAST_OUT"

# rc 75: hold the lock so flock times out. Not one assertion runs.
#
# The holder must ACTUALLY hold the lock before the gate runs. A fixed `sleep 0.5` raced on
# slow/loaded runners (observed 2026-09-01 in GitHub CI: the gate won the lock, went red for
# an unrelated reason, and the assertion below reported "pattern not found: CANARY
# INCONCLUSIVE .* rc 75" — a flaky RED that says nothing about the property under test).
# Poll for the holder's own readiness flag instead, and fail loudly if it never arms.
( flock 9 || exit 1; : > "$WORK/holder.armed"; sleep 30 ) 9>"$WORK/gpu.lock" &
HOLDER=$!
echo "$HOLDER" > "$WORK/holder.pid"
for _ in $(seq 1 100); do
    [ -e "$WORK/holder.armed" ] && break
    sleep 0.1
done
if [ ! -e "$WORK/holder.armed" ]; then
    echo "FIXTURE BROKEN: the lock holder never armed in 10s — the rc 75 arm cannot be tested" >&2
    kill "$HOLDER" 2>/dev/null
    exit 1
fi
run_gate "MEMRA_GPU_LOCK_WAIT=1" -- --canary
assert_rc "canary: rc 75 (lock timeout) is INCONCLUSIVE, never OK" 1 "$LAST_RC"
assert_grep "canary: rc 75 is named as the lock timeout" \
    'CANARY INCONCLUSIVE .* rc 75' "$LAST_OUT"
kill "$HOLDER" 2>/dev/null
wait "$HOLDER" 2>/dev/null
rm -f "$WORK/holder.pid" "$WORK/holder.armed"

# ---------------------------------------------------------------------------
# SKIP CENSUS (Item 2). The tool is exercised against throwaway crates and a stub cargo, so the
# arms are decisive without a Rust build.
# ---------------------------------------------------------------------------
CENSUS_TOOL="$GATE_SRC/tools/skip-census.py"
if [ ! -f "$CENSUS_TOOL" ]; then
    for name in \
        "skip census: verify agrees with the source" \
        "skip census: an UNDECLARED skipping test fails verify" \
        "skip census: a STALE manifest row fails verify" \
        "skip census: skips over budget FAIL and name the budget variable" \
        "skip census: an explicitly raised budget passes and reports the count" \
        "skip census: a red suite fails on the suite verdict before any skip count" \
        "skip census: a name-filtered suite is refused" \
        "skip census: an uninitialised census file is a wiring failure, not zero skips" \
        "skip census: the file count is an EQUALITY"; do
        fail "$name" "tools/skip-census.py does not exist in $GATE_SRC"
    done
else
    CENSUS_OUT="$WORK/census.out"
    ( cd "$GATE_SRC" && python3 tools/skip-census.py verify ) > "$CENSUS_OUT" 2>&1
    assert_rc "skip census: verify agrees with the source" 0 $?

    # An undeclared skipping test: add one to a COPY of the crate tree.
    CRATE_COPY="$WORK/crate-copy"
    mkdir -p "$CRATE_COPY/tools" "$CRATE_COPY/crates/memra-gguf/src"
    cp "$CENSUS_TOOL" "$CRATE_COPY/tools/"
    cp "$GATE_SRC/tools/skip-census.tsv" "$CRATE_COPY/tools/" 2>/dev/null || true
    cat > "$CRATE_COPY/crates/memra-gguf/src/source.rs" <<'RS'
#[cfg(test)]
mod probe {
    #[test]
    fn a_brand_new_artifact_gated_test() {
        if !std::path::Path::new("/nope").exists() {
            eprintln!("SKIP: brand new blind spot");
            return;
        }
    }
}
RS
    UND="$WORK/undeclared.out"
    ( cd "$CRATE_COPY" && python3 tools/skip-census.py verify ) > "$UND" 2>&1
    UND_RC=$?
    if [ "$UND_RC" -ne 0 ] && grep -q 'a_brand_new_artifact_gated_test' "$UND"; then
        pass "skip census: an UNDECLARED skipping test fails verify"
    else
        fail "skip census: an UNDECLARED skipping test fails verify" "rc=$UND_RC"
    fi
    if grep -q 'no longer exists in the source' "$UND"; then
        pass "skip census: a STALE manifest row fails verify"
    else
        fail "skip census: a STALE manifest row fails verify" "no stale-row diagnosis in $UND"
    fi

    # Stub cargo: canned libtest output, so the run arms need no build.
    CARGO_HOME_STUB="$WORK/cargohome"
    mkdir -p "$CARGO_HOME_STUB/bin"
    cat > "$CARGO_HOME_STUB/bin/cargo" <<'SH'
#!/usr/bin/env bash
case "${FIXTURE_CARGO_MODE:-skips}" in
  skips)
    printf 'running 4 tests\n'
    printf 'test source::nv27b_twin_parity::nvidia_27b_vs_gguf_twin_f32_parity ... SKIP: ckpt/twin absent\nok\n'
    printf 'test source::m3_probe::minimax_m3_lm_head_q8 ... SKIP: ckpt absent\nok\n'
    printf 'test a::b ... ok\n'
    printf 'test c::d ... ok\n'
    printf 'test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n' ;;
  red)
    printf 'test a::b ... FAILED\n'
    printf 'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n'
    exit 101 ;;
  filtered)
    printf 'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s\n' ;;
  undeclared)
    printf 'test source::probe::brand_new ... SKIP: something nobody declared\nok\n'
    printf 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n' ;;
esac
SH
    chmod +x "$CARGO_HOME_STUB/bin/cargo"

    census_run() { # $1 mode  $2 budget  -> LAST_RC/LAST_OUT
        LAST_OUT="$WORK/census-run-$1-$2.out"
        ( cd "$GATE_SRC" && PATH="$CARGO_HOME_STUB/bin:$PATH" FIXTURE_CARGO_MODE="$1" \
            MEMRA_GGUF_SKIP_BUDGET="$2" python3 tools/skip-census.py run \
            --budget-var MEMRA_GGUF_SKIP_BUDGET --min-passed 1 \
            -- cargo test -p memra-gguf --lib ) > "$LAST_OUT" 2>&1
        LAST_RC=$?
    }

    census_run skips 0
    if [ "$LAST_RC" -ne 0 ] && grep -q 'budget 0 (MEMRA_GGUF_SKIP_BUDGET)' "$LAST_OUT"; then
        pass "skip census: skips over budget FAIL and name the budget variable"
    else
        fail "skip census: skips over budget FAIL and name the budget variable" "rc=$LAST_RC"
    fi
    census_run skips 2
    if [ "$LAST_RC" -eq 0 ] && grep -q '2 skipped (budget 2)' "$LAST_OUT"; then
        pass "skip census: an explicitly raised budget passes and reports the count"
    else
        fail "skip census: an explicitly raised budget passes and reports the count" "rc=$LAST_RC"
    fi
    census_run red 99
    if [ "$LAST_RC" -ne 0 ] && grep -q 'the suite exited' "$LAST_OUT"; then
        pass "skip census: a red suite fails on the suite verdict before any skip count"
    else
        fail "skip census: a red suite fails on the suite verdict before any skip count" "rc=$LAST_RC"
    fi
    census_run filtered 99
    if [ "$LAST_RC" -ne 0 ] && grep -q 'FILTERED OUT of an unfiltered run' "$LAST_OUT"; then
        pass "skip census: a name-filtered suite is refused"
    else
        fail "skip census: a name-filtered suite is refused" "rc=$LAST_RC"
    fi

    RPT="$WORK/report.out"
    ( cd "$GATE_SRC" && python3 tools/skip-census.py report "$WORK/never-made.tsv" --expect 0 ) \
        > "$RPT" 2>&1
    RPT_RC=$?
    if [ "$RPT_RC" -ne 0 ] && grep -q 'never initialised' "$RPT"; then
        pass "skip census: an uninitialised census file is a wiring failure, not zero skips"
    else
        fail "skip census: an uninitialised census file is a wiring failure, not zero skips" \
            "rc=$RPT_RC"
    fi
    ( cd "$GATE_SRC" && python3 tools/skip-census.py report "$CENSUS" --expect 5 ) \
        > "$WORK/report2.out" 2>&1
    RPT2_RC=$?
    if [ "$RPT2_RC" -ne 0 ] && grep -q 'An EQUALITY, not a ceiling' "$WORK/report2.out"; then
        pass "skip census: the file count is an EQUALITY"
    else
        fail "skip census: the file count is an EQUALITY" "rc=$RPT2_RC"
    fi
fi

# ---------------------------------------------------------------------------
# Verdict. The count is an equality, and a miss is a BROKEN FIXTURE, not a smaller green run.
# ---------------------------------------------------------------------------
TOTAL=$(grep -c . "$VERDICTS")
PASSED=$(grep -c '^PASS' "$VERDICTS")
FAILED=$(grep -c '^FAIL' "$VERDICTS")
echo
echo "=== gate-template integrity fixture: $PASSED passed / $FAILED failed of $TOTAL ==="
if [ "$GEN_MODE" = v1 ]; then
    echo "NOTE: the generator in $GATE_SRC rejected batch.canary_expect_regex, so this run used"
    echo "  the v1 spec (port 8094). Arms about the schema are non-decisive here BY CONSTRUCTION;"
    echo "  the artifact arms are the decisive ones."
fi
if [ "$TOTAL" -ne "$EXPECT_ASSERTIONS" ]; then
    echo "BROKEN FIXTURE: $TOTAL assertions recorded, $EXPECT_ASSERTIONS declared." >&2
    echo "  An arm that stops running must red this fixture, not make it smaller." >&2
    exit 3
fi
[ "$FAILED" -eq 0 ] || exit 1
echo "gate-template integrity: ALL $TOTAL ASSERTIONS GREEN"
