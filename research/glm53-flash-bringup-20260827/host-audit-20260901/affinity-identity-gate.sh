#!/usr/bin/env bash
# BYTE-IDENTITY GATE for MEMRA_WORKER_AFFINITY (lane/glm5-host-audit, 2026-09-01).
#
# THE CLAIM UNDER TEST: pinning the GPU worker thread to a CPU mask changes no bytes. That is
# "obviously" true — a CPU mask selects no kernel, reorders no reduction, and moves no tolerance —
# and "obviously non-numeric" is exactly how a numeric-class door ships unmeasured, so it is
# asserted rather than argued. ENGINE-WIDE flag ⇒ the gate runs on MORE THAN ONE FAMILY.
#
# RIG LAW (rig-gpu-exactness-only): the 5090 laptop throttles to ~52% clock, so this script
# produces IDENTITY receipts and NEVER a timing number. It takes the same `/tmp/memra-5090.lock`
# local-ci uses, fd-based, so it serializes against every other rig GPU consumer.
#
# WHAT IT PROVES, per family:
#   1. OFF arm and ON arm emit BYTE-IDENTICAL text over a fixed 24-step greedy generation
#      (greedy = the instrument: it is the only decoding that makes a byte oracle possible).
#   2. The ON boot ANNOUNCES with a kernel readback (`[worker-affinity] engaged … effective=…`)
#      and the OFF boot announces `off` — arm identity from the server's own log, never from a
#      200 on /readyz (LAW ab-arm-identity-not-liveness).
#   3. The effective mask is NARROWER than the machine (i.e. the pin actually took), read back
#      from the announce line rather than assumed from the request.
#
# WHAT IT DOES NOT COVER, stated rather than implied: glm5. No full GLM-5.3-Flash artifact exists
# on the rig (only the 1.2 GB vision tower), so the glm5 arm of this gate MUST run in the Box B
# window against the real NVFP4 artifact, and LANE.md's gate table carries it as PENDING until it
# does. Do not read a green run here as covering the hybrid residual topology.
#
# Usage:
#   bash affinity-identity-gate.sh                       # every family it can find on this host
#   MODELS="qwen=$HOME/models/foo.gguf" bash affinity-identity-gate.sh
set -uo pipefail

LOCK="${MEMRA_CI_LOCK:-/tmp/memra-5090.lock}"
LOCK_WAIT="${LOCK_WAIT:-3600}"
BIN="${BIN:-$(cd "$(dirname "$0")/../../.." && pwd)/target/release/memra-server}"
OUT="${OUT:-${TMPDIR:-/tmp}/ha-identity-$$}"
PORT="${PORT:-18711}"
STEPS="${STEPS:-24}"   # 24 steps: the brief's bar
PROMPT="${PROMPT:-Write one short paragraph explaining what a CPU cache is.}"

mkdir -p "$OUT"
# Scratch is cleaned by the task that created it (owner hygiene rule), including on failure.
trap 'stop_server; rm -rf "$OUT"' EXIT

fail() { echo "GATE FAIL: $*" >&2; exit 1; }
# STDERR, deliberately. `run_arm`'s stdout IS the sha — it is read through command
# substitution — so a progress line on stdout would be captured into the "sha" and, because
# every note contains the arm name, the OFF and ON captures would differ by construction and
# this gate would report a CONFIDENT BYTE DIVERGENCE that does not exist. Caught by reading the
# capture path rather than by a run, which is the only way this class is ever caught: a
# false RED looks exactly like a real finding.
note() { echo "[$(date -u +%FT%TZ)] $*" >&2; }

PIDFILE="$OUT/server.pid"

stop_server() {
    [ -f "$PIDFILE" ] || return 0
    local pid exe
    pid=$(cat "$PIDFILE" 2>/dev/null || echo "")
    if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
        exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || echo "")
        # PID + exe verified, never pkill and never a basename kill
        # (LAW gate-stop-pkill-basename-trap: a renamed comparison binary orphans a
        # VRAM-holding server and corrupts the very oracle the gate exists to produce).
        if [ "$exe" = "$(readlink -f "$BIN")" ]; then
            kill "$pid" 2>/dev/null || true
            for _ in $(seq 1 60); do [ -d "/proc/$pid" ] || break; sleep 1; done
            [ -d "/proc/$pid" ] && kill -9 "$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$PIDFILE"
}

# One boot. Echoes the sha256 of the emitted text on stdout; the log lands in $OUT.
run_arm() {
    local family="$1" model="$2" arm="$3" affinity="$4"
    # A model alias is `vendor/name` — it contains a SLASH, so it cannot go into a filename
    # unescaped. First run of this gate did exactly that: the `>"$log"` redirect pointed into a
    # directory that does not exist, the server therefore never started, and the failure surfaced
    # as "never became ready" — a harness bug wearing the costume of a real identity failure.
    # The gate was loud (it went RED and named the missing path), which is the only reason this
    # was five minutes of work instead of a false finding.
    local slug="${family//\//_}"
    local log="$OUT/serve-$slug-$arm.log"
    stop_server
    local env_extra=()
    [ -n "$affinity" ] && env_extra+=("MEMRA_WORKER_AFFINITY=$affinity")
    ( env MEMRA_COMPAT=openai "MEMRA_MODELS=$family=$model" \
          "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=4096 MEMRA_MAX_SESSIONS=2 \
          NVIDIA_TF32_OVERRIDE=0 "${env_extra[@]}" \
          "$BIN" >"$log" 2>&1 & echo $! >"$PIDFILE" )
    local pid; pid=$(cat "$PIDFILE")
    local ready=0
    for _ in $(seq 1 600); do
        curl -sf "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1 && { ready=1; break; }
        [ -d "/proc/$pid" ] || break
        sleep 1
    done
    [ "$ready" = 1 ] || { tail -25 "$log" >&2; fail "$family/$arm never became ready"; }

    # ARM IDENTITY from the log, not from liveness.
    local line; line=$(grep -m1 '\[worker-affinity\]' "$log" || true)
    [ -n "$line" ] || fail "$family/$arm printed no [worker-affinity] line (both arms must announce)"
    note "$family/$arm: $line"
    if [ -n "$affinity" ]; then
        echo "$line" | grep -q 'engaged .*effective=' \
            || fail "$family/$arm: ON arm did not announce an engaged+readback line: $line"
        # The readback must be NARROWER than the machine, or the pin did not take.
        local eff ncpu
        eff=$(echo "$line" | grep -o 'cpus=[0-9]*' | cut -d= -f2)
        ncpu=$(nproc --all)
        [ -n "$eff" ] || fail "$family/$arm: no cpus= in the announce"
        [ "$eff" -lt "$ncpu" ] \
            || fail "$family/$arm: effective mask is $eff cpus of $ncpu — the pin did NOT narrow anything, so this is an OFF-arm boot wearing an ON-arm label"
        note "$family/$arm: effective mask narrowed to $eff of $ncpu cpus"
    else
        echo "$line" | grep -q '\[worker-affinity\] off' \
            || fail "$family/$arm: OFF arm announced something else: $line"
    fi

    # 24 greedy steps. temperature 0 = the byte-deterministic instrument.
    local body resp text
    body=$(python3 - "$family" "$PROMPT" "$STEPS" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "max_tokens": int(sys.argv[3]),
    "temperature": 0.0,
    "stream": False,
    "reasoning_effort": "low",
}))
PY
)
    resp=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body") \
        || { tail -25 "$log" >&2; fail "$family/$arm request failed"; }
    # Hash the reasoning AND the content, in that order. A THINKING model at a small token cap
    # emits only reasoning and leaves `content` empty (the qwen 9B judge fixture does exactly
    # this), so hashing `content` alone compared "" against "" — a vacuous pass that the
    # emptiness guard below caught. reasoning_effort is pinned in the body for the same reason
    # the decode cells pin it (TRAP reasoning-effort-unpinned-decode-cell).
    text=$(printf '%s' "$resp" | python3 -c 'import json,sys
d = json.load(sys.stdin)
m = d["choices"][0]["message"]
# The OpenAI-compat surface names this field `reasoning` (verified against a live
# response: {"content":"","reasoning":"Thinking Process:..."}). `reasoning_content` is the
# name the other surfaces use, so both are read rather than guessed at.
print((m.get("reasoning") or m.get("reasoning_content") or "") + (m.get("content") or ""), end="")')
    [ -n "$text" ] || fail "$family/$arm produced EMPTY text (content AND reasoning) — an empty-vs-empty comparison is a vacuous pass"
    printf '%s' "$text" > "$OUT/text-$slug-$arm.txt"
    printf '%s' "$text" | sha256sum | cut -c1-16
}

[ -x "$BIN" ] || fail "no memra-server at $BIN (build it first: cargo build --release -p memra-server)"

# Families discovered on this host, not hardcoded.
declare -a FAMILIES=()
if [ -n "${MODELS:-}" ]; then
    for entry in $MODELS; do FAMILIES+=("$entry"); done
else
    [ -f "$HOME/models/qwen3.5-9b-judge-q8_0.gguf" ] && \
        FAMILIES+=("qwen/qwen3.5-9b=$HOME/models/qwen3.5-9b-judge-q8_0.gguf")
    # The ornith fixture is a GGUF TREE, not an HF safetensors dir: the server refuses the
    # directory ("want model.safetensors ... or manifest.json"), so name the .gguf itself.
    for f in "$HOME/models/ornith15-9b/gguf/Ornith-1.5-9B-NVFP4-Q5K-mtp.gguf"; do
        [ -f "$f" ] && FAMILIES+=("ornith/ornith-1.5-9b=$f")
    done
fi
[ ${#FAMILIES[@]} -gt 0 ] || fail "no fixture families found — pass MODELS=\"alias=path\""
note "families under test: ${FAMILIES[*]%%=*}"
note "NOT COVERED HERE: glm5 (no full artifact on this host) — its arm runs in the Box B window"

# THE ON-ARM MASK, chosen from THIS host's real L3 map rather than assumed.
#
# `ccx` is the right form on a multi-CCD server, but it is MEANINGLESS on a host with a single
# L3 domain: this rig is a 24-thread laptop part whose one domain is `0-23`, so `ccx` resolved to
# the whole machine and the mask narrowed NOTHING. The gate caught it ("effective mask is 24 cpus
# of 24 ... an OFF-arm boot wearing an ON-arm label") — which is the assertion earning its keep,
# because a full-width mask would otherwise have produced a cheerful green that proved only that
# two unpinned boots agree.
#
# So: use `ccx` when it genuinely narrows, and otherwise fall back to an EXPLICIT sub-machine
# list. Byte identity under a real narrowing mask is the claim; which spelling produced the mask
# is not.
CCX_LIST=$(cat /sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list 2>/dev/null || echo "")
NCPU=$(nproc --all)
ON_MASK="ccx"
if [ -n "$CCX_LIST" ]; then
    ccx_n=$(python3 -c '
import sys
raw = sys.argv[1]
n = 0
for part in raw.split(","):
    if "-" in part:
        lo, hi = part.split("-")
        n += int(hi) - int(lo) + 1
    elif part.strip():
        n += 1
print(n)' "$CCX_LIST")
    if [ "$ccx_n" -ge "$NCPU" ]; then
        # Single-L3 host: quarter of the machine, at least 2 cpus, as an explicit list.
        half=$(( NCPU / 4 )); [ "$half" -lt 2 ] && half=2
        ON_MASK="0-$(( half - 1 ))"
        note "single-L3 host ($CCX_LIST covers all $NCPU cpus): 'ccx' would not narrow, so the ON arm uses the explicit list $ON_MASK"
    else
        note "multi-L3 host: the ON arm uses the 'ccx' form (domain $CCX_LIST)"
    fi
else
    note "WARN: no index3 in sysfs; the ON arm will use the 'ccx' form and is expected to REFUSE"
fi

exec 9>"$LOCK"
flock -w "$LOCK_WAIT" 9 || fail "could not take the rig GPU lock $LOCK within ${LOCK_WAIT}s"
note "rig GPU lock held (exactness only — this gate never reports a timing number)"

rc=0
for entry in "${FAMILIES[@]}"; do
    family="${entry%%=*}"; model="${entry#*=}"
    note "=== $family ==="
    off=$(run_arm "$family" "$model" off "")   || { rc=1; continue; }
    on=$(run_arm  "$family" "$model" on  "$ON_MASK")  || { rc=1; continue; }
    # The captures must be sha16 and NOTHING else. Asserted rather than assumed: if a stray
    # progress line ever reaches run_arm's stdout again, this says so instead of reporting the
    # contamination as a byte divergence.
    for got in "$off" "$on"; do
        [[ "$got" =~ ^[0-9a-f]{16}$ ]] \
            || fail "$family: captured '$got' is not a bare sha16 — run_arm's stdout is contaminated, so a comparison here would be meaningless"
    done
    if [ "$off" = "$on" ]; then
        note "PASS $family: ${STEPS}-step greedy BYTE-IDENTICAL across the affinity arms (sha16 $off)"
    else
        note "FAIL $family: sha16 OFF=$off ON=$on"
        diff <(cat "$OUT/text-${family//\//_}-off.txt") <(cat "$OUT/text-${family//\//_}-on.txt") | head -20 >&2 || true
        rc=1
    fi
done
stop_server
[ "$rc" = 0 ] && note "GATE GREEN: affinity is non-numeric on every family tested" \
             || note "GATE RED"
exit "$rc"
