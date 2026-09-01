#!/usr/bin/env bash
# spec-cache-gate.sh — the lane/spec-prefix-cache headline gate (GPU box).
#
# Proves the commit-gated publication port on the SOLD SHAPE (canonflip
# 2026-08-13 protocol): ~4,860-token shared prefix + unique tails, 60 output
# tokens, 10% forced-miss salts, c=16, N=3 passes per arm.
#
#   PASS iff, per pass medians:
#     1. automatic-policy req/s >= 0.9 x spec-off req/s (best-policy throughput recovered)
#     2. both spec-on cache-hit rates >= 0.95            (probe + publication live;
#        hit rate = sum cached_tokens / sum prompt_tokens over non-miss requests)
#     3. EXACTNESS ANCHOR: the greedy anchor prompt's sha256(reasoning+content)
#        identical across ALL arms and passes (temp 0, fresh salt per pass).
#
# Two spec-on sub-cells:
#   a. production config (default spec gate — measures the real composition)
#   b. MEMRA_SPEC_K pinned to 3 (diagnostic: proves a pin cannot suppress publication;
#      its forced serial policy is recorded but is not a production throughput candidate)
#
# SAMPLED SOLD-SHAPE ARM (lane/sampled-hit-spec §7 item 3, added in the 96GB window).
# MEMRA_CACHEGATE_TEMP > 0 fires the MEASURED load rows at that temperature — the traffic
# shape that actually pays, since the OpenAI surface defaults to temperature 1.0. Why it
# had to be added rather than assumed: every row above was hardcoded `temperature: 0`, so
# the sold-shape headline (~135 -> ~75 tok/s on cache-hit rows) had only ever been measured
# on greedy traffic, while v0.93.0's restore was greedy-ONLY and therefore inert for
# exactly the rows this gate was quoting. The lane measured 1.210x on a 9B / 106-token
# shape; this arm is the prod-sized read.
#
# The exactness ANCHOR stays at temperature 0 in every arm, deliberately: it is a byte
# identity check, and sampled spec is distributionally exact rather than byte-equal to
# plain sampling (see tools/spec-on-cache-hit-gate.sh), so a sampled anchor would be a
# fake gate. Sampled rows carry a per-row `seed` so a pass is reproducible.
#
# When MEMRA_CACHEGATE_TEMP > 0 two more spec-on sub-cells run:
#   c. on-nosampled = MEMRA_SPEC_RESTORE_SAMPLED=0 — the v0.93.0 posture, in which every
#      sampled cache hit downgrades to plain. on-gate vs on-nosampled is the LEVER's
#      payoff on the sold shape, isolated (same boot config otherwise).
#   d. on-noguard = MEMRA_SPEC_RESTORE_LOAD_GUARD=0 (added 2026-08-19,
#      lane/sampled-restore-load-guard) — the lever with its LOAD GUARD DISABLED, i.e. the
#      posture the 96GB window measured at 0.809x on this exact cell. It is the arm that
#      makes the guard's effect attributable inside one protocol: off / lever-off /
#      lever-on-unguarded / lever-on-guarded, same box, same binary, same interleave.
#
# CONCURRENCY IS A KNOB NOW (MEMRA_CACHEGATE_CONC, default 16 = the sold shape). The load
# guard's whole claim is that the lever's sign FLIPS with concurrency, so the instrument that
# measures the sign has to be able to walk the ladder. Sweeping this variable across
# 1,2,3,4,8,16 with the arms below IS the crossover measurement — no separate tool, so the
# crossover and the headline cell are the same protocol and are comparable by construction.
#
# VERDICT RULE, pre-registered before the run (2026-08-19 revision):
#   floors (both regimes): on-gate req/s >= 0.9x off; both spec-on hit rates >= 0.95;
#                          anchor sha identical across ALL arms and passes.
#   sampled regime, and this is the part the load guard changed: the rule is keyed on the
#   SERVER'S OWN policy, derived from its `[spec-gate] policy ... LOW=n HIGH=m` boot line
#   (watermark = min(LOW, 1) = SOLO, matching worker.rs `sampled_restore_watermark`), because
#   "what should happen here" is a function of the policy, not of a number in this file.
#     * ABOVE the watermark (CONC > the sampled-restore watermark — the guard must refuse):
#         on-gate engaged rows == 0, the `[spec-restore-guard] ... REFUSED` line PRESENT
#         (a silent non-refusal is a FAIL, not a pass — the v0.93.0 lesson), and PARITY:
#         on-gate / on-nosampled within [0.98, 1.02]. Refusing and still losing throughput
#         would mean the guard is not the whole story.
#         `on-noguard` is recorded and reported but does NOT gate: it is the control that
#         shows the loss is still there with the door open.
#     * AT OR BELOW the watermark (the guard must admit):
#         on-gate engaged rows > 0 AND recovery = on-gate / on-nosampled >= 1.05, i.e. the
#         original rule, unchanged, in the tier where the lever pays.
#
# usage: spec-cache-gate.sh <model_path> <server_bin> <evidence_dir>
#   MEMRA_CACHEGATE_CONC=<n>      load concurrency (default 16)
#   MEMRA_CACHEGATE_REQS=<n>      total measured requests (default = CONC = the banked one-burst
#                                 shape). > CONC runs a closed-loop pool of CONC workers, i.e.
#                                 SUSTAINED saturation — the shape whose aggregate is not
#                                 dominated by the wave's own head.
#   MEMRA_CACHEGATE_TEMP=<t>      measured-row temperature (default 0 = greedy)
#   MEMRA_CACHEGATE_PASSES=<n>    interleaved passes (default 3)
#   MEMRA_GATE_MTP_DRAFT=<f>      attach the real external MTP drafter and ASSERT the attach
# Boots its own servers (flock /tmp/memra-5090.lock by default), one arm at a time,
# interleaved pass-major: off,a,b / off,a,b / off,a,b.
set -euo pipefail
MODEL=$1; BIN=$2; EV=$3
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_GATE_PORT:-18086}
HERE=$(cd "$(dirname "$0")" && pwd)
# Port occupancy guard (GATE-INTEGRITY-20260819 A-16, deferred to this file's merge because
# the sampled lanes had it open). An occupied port is a HARD ABORT before every boot: a
# foreign responder that speaks the API would be measured and pinned. No memra_port_owned —
# $SERVER_PID here is the flock wrapper's pid, and pid equality would manufacture a false red
# (the dspark-serve-smoke precedent).
[ -f "$HERE/port-guard.sh" ] || {
    echo "spec-cache-gate: FAIL — $HERE/port-guard.sh is missing; refusing to bind unguarded" >&2
    exit 1
}
. "$HERE/port-guard.sh"
PREFIX_TOKENS=4860
SUFFIX_TOKENS=64
OUT_TOKENS=60
CONC=${MEMRA_CACHEGATE_CONC:-16}
# Total measured requests. Default = CONC (the banked one-burst shape, unchanged);
# > CONC runs a closed-loop pool of CONC workers so saturation outlasts its own transient.
REQS=${MEMRA_CACHEGATE_REQS:-$CONC}
PASSES=${MEMRA_CACHEGATE_PASSES:-3}
MISS_PCT=10
CACHE_TEMP=${MEMRA_CACHEGATE_TEMP:-0}
# The QWEN attach is MEMRA_MTP_DRAFT; MEMRA_DRAFT is the GEMMA assistant-drafter seam and on a
# qwen model attaches NOTHING while still flipping wkv_on()/fa_f16pv_on()/the MMQ-SK form
# (docs/FLAGS.md 'SEAM TRAP'). When set, every boot asserts the attach line.
MTP_DRAFT=${MEMRA_GATE_MTP_DRAFT:-}
DRAFT_FAILS=0
mkdir -p "$EV"

boot() { # $1 extra-env  $2 log
    memra_port_guard spec-cache-gate "$PORT" MEMRA_GATE_PORT || return 1
    local mtp=()
    [ -n "$MTP_DRAFT" ] && mtp=("MEMRA_MTP_DRAFT=$MTP_DRAFT")
    # shellcheck disable=SC2086
    flock -w 120 "$GPU_LOCK" env CUDA_VISIBLE_DEVICES=0 \
        MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=$((CONC + 2)) \
        "${mtp[@]}" $1 "$BIN" > "$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        if curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            if [ -n "$MTP_DRAFT" ] \
                && ! "$HERE/assert-drafter-attached.sh" "$2" "$MTP_DRAFT"; then
                DRAFT_FAILS=$((DRAFT_FAILS + 1))
            fi
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null \
            || { echo "server died"; wait "$SERVER_PID" || true; tail -5 "$2"; return 1; }
        sleep 5
    done
    return 1
}
stop() {
    pkill -x memra-server 2>/dev/null || true
    for _ in $(seq 1 30); do
        curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || return 0
        sleep 2
    done
    pkill -9 -x memra-server 2>/dev/null || true; sleep 3
}

run_pass() { # $1 arm  $2 pass  -> writes $EV/<arm>-p<pass>.json summary
    python3 - "$PORT" "$1" "$2" "$EV" "$PREFIX_TOKENS" "$SUFFIX_TOKENS" \
        "$OUT_TOKENS" "$CONC" "$MISS_PCT" "$CACHE_TEMP" "$REQS" <<'PY'
import hashlib, itertools, json, sys, threading, time, urllib.request
port, arm, pss, ev, ptoks, stoks, otoks, conc, miss_pct, load_temp, reqs = (
    int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4],
    int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]),
    int(sys.argv[9]), float(sys.argv[10]), int(sys.argv[11]))
rows, lock = [], threading.Lock()
def fire(i, forced_miss=None, shared_prefix_only=False):
    if forced_miss is None:
        forced_miss = (i % (100 // miss_pct)) == 0 if miss_pct else False
    salt = f"gate-{arm}-p{pss}" + (f"-miss{i}" if forced_miss else "")
    ids = [11] * ptoks
    if not shared_prefix_only:
        ids += [1000 + (i * 37 + j) % 4000 for j in range(stoks)]
    body = {"model": "gate", "prompt_ids": ids, "max_tokens": otoks,
            "temperature": load_temp, "cache_salt": salt}
    # A sampled row gets an explicit per-row seed: an OMITTED seed is fresh entropy per
    # request, which would make a pass unreproducible for no measurement gain.
    if load_temp > 0:
        body["seed"] = 90000 + i
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); row = {"i": i, "ok": False, "miss_arm": forced_miss}
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            b = json.load(r)
        u = b.get("usage") or {}
        sp = u.get("spec") or {}
        row.update(ok=True,
                   prompt_tokens=u.get("prompt_tokens", 0),
                   cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                   completion=u.get("completion_tokens", 0),
                   spec_accepted=sp.get("accepted"),
                   spec_drafted=sp.get("drafted"))
    except Exception as e:
        row["err"] = repr(e)[:120]
    row["e2e"] = round(time.monotonic() - t0, 3)
    with lock: rows.append(row)
# Warm the exact common-prefix boundary in the same namespace the measured
# non-miss requests use. A full request includes a unique tail, and immediate
# partial restore is intentionally disabled, so publishing that longer prompt
# cannot serve the shared prefix needed by the measured burst.
fire(0, forced_miss=False, shared_prefix_only=True)
t0 = time.monotonic()
if reqs <= conc:
    # BANKED SHAPE (reqs == conc): one burst of `conc` threads. Byte-identically the protocol
    # every earlier receipt was taken under.
    ts = [threading.Thread(target=fire, args=(i,)) for i in range(1, conc + 1)]
    [t.start() for t in ts]; [t.join() for t in ts]
else:
    # SUSTAINED SHAPE (MEMRA_CACHEGATE_REQS > conc): a closed-loop pool of `conc` workers
    # serving `reqs` requests, i.e. saturation that OUTLASTS its own head. Why it exists: the
    # burst shape's aggregate is dominated by its transient. A wave arrives at an idle box, so
    # its head is admitted while the box genuinely IS quiet, and one spec row inside an
    # exact-16 wave costs ~9.6% of that wave's throughput — on a 16-request pass that single
    # row IS the headline number. Production saturation is not one burst; measuring both says
    # which part of a loss is transient and which is steady-state.
    nxt = itertools.count(1)
    def worker():
        while True:
            i = next(nxt)
            if i > reqs:
                return
            fire(i)
    ts = [threading.Thread(target=worker) for _ in range(conc)]
    [t.start() for t in ts]; [t.join() for t in ts]
wall = time.monotonic() - t0
ok = [r for r in rows[1:] if r["ok"]]
shared = [r for r in ok if not r["miss_arm"]]
hit_rate = (sum(r["cached"] for r in shared) / max(1, sum(r["prompt_tokens"] for r in shared)))
# exactness anchor: fixed short prompt, fresh salt, greedy
anchor_req = {"model": "gate",
              "messages": [{"role": "user", "content": "List three prime numbers."}],
              "max_tokens": 48, "temperature": 0, "cache_salt": f"anchor-{arm}-p{pss}"}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(anchor_req).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    m = (json.load(r).get("choices") or [{}])[0].get("message", {})
anchor = hashlib.sha256(((m.get("reasoning") or "") + "\x1f" + (m.get("content") or "")).encode()).hexdigest()
engaged = [r for r in ok if (r.get("spec_accepted") or 0) > 0]
summ = {"arm": arm, "pass": pss, "ok": len(ok), "conc": conc, "reqs": reqs,
        "req_s": round(len(ok) / wall, 3),
        "hit_rate": round(hit_rate, 4), "anchor": anchor, "wall_s": round(wall, 2),
        "load_temp": load_temp, "engaged": len(engaged),
        "accepted": sum((r.get("spec_accepted") or 0) for r in ok),
        "drafted": sum((r.get("spec_drafted") or 0) for r in ok)}
with open(f"{ev}/{arm}-p{pss}.json", "w") as f: json.dump({"rows": rows, "summary": summ}, f)
print(json.dumps(summ))
PY
}

declare -A ENVS=(
    [off]="MEMRA_SERVE_SPEC=0"
    [on-gate]=""
    [on-pin]="MEMRA_SPEC_K=3"
    # v0.93.0 posture: sampled cache hits downgrade to plain. Only meaningful when the
    # measured rows are sampled, so it is added to the arm list only in that regime.
    [on-nosampled]="MEMRA_SPEC_RESTORE_SAMPLED=0"
    # the lever with its load guard OPEN — the 0.809x posture, kept as the control
    [on-noguard]="MEMRA_SPEC_RESTORE_LOAD_GUARD=0"
)
ARMS="off on-gate on-pin"
awk_gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>0 else 1)" "$1"; }
if awk_gt "$CACHE_TEMP"; then ARMS="off on-gate on-pin on-nosampled on-noguard"; fi
# MEMRA_CACHEGATE_ARMS overrides the list (a crossover sweep wants only the two arms whose
# ratio IS the crossover, and paying for on-pin at six concurrencies buys nothing).
ARMS=${MEMRA_CACHEGATE_ARMS:-$ARMS}
echo "spec-cache-gate: c$CONC, $REQS requests, load temperature $CACHE_TEMP, \
passes $PASSES, arms: $ARMS"
for p in $(seq 1 "$PASSES"); do
    for arm in $ARMS; do
        echo "== pass $p arm $arm =="
        stop
        boot "${ENVS[$arm]}" "$EV/server-$arm-p$p.log" || exit 1
        run_pass "$arm" "$p"
    done
done
stop

python3 - "$EV" "$PASSES" "$CONC" "$DRAFT_FAILS" "$ARMS" <<'PY'
import json, re, statistics, sys
ev, passes, conc, draft_fails = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                 int(sys.argv[4]))
arms = sys.argv[5].split()
def summ(arm, p):
    return json.load(open(f"{ev}/{arm}-p{p}.json"))["summary"]
def med(arm, key):
    return statistics.median(summ(arm, p)[key] for p in range(1, passes + 1))
def log(arm, p):
    try:
        return open(f"{ev}/server-{arm}-p{p}.log", errors="replace").read()
    except FileNotFoundError:
        return ""
anchors = set()
for arm in arms:
    for p in range(1, passes + 1):
        anchors.add(summ(arm, p)["anchor"])
off_rs = med("off", "req_s")
temp = summ("on-gate", 1)["load_temp"]
fails = []
if draft_fails:
    fails.append(f"{draft_fails} drafter-attach assertion(s) — a drafter was handed to this "
                 "gate and never loaded, so every row above measured the wrong config")
for arm in [a for a in arms if a != "off"]:
    rs, hr = med(arm, "req_s"), med(arm, "hit_rate")
    eng = [summ(arm, p)["engaged"] for p in range(1, passes + 1)]
    print(f"{arm}: req/s {rs} ({rs / off_rs:.3f}x off {off_rs}), hit {hr}, "
          f"engaged rows/pass {eng}")
    if arm == "on-gate" and rs < 0.9 * off_rs:
        fails.append(f"{arm} req/s {rs} < 0.9x off {off_rs}")
    if hr < 0.95: fails.append(f"{arm} hit rate {hr} < 0.95")
if len(anchors) != 1: fails.append(f"exactness anchor diverged: {len(anchors)} distinct")

# ---- Sampled regime. The rule is keyed on the SERVER'S OWN band (see header). ----
if temp > 0 and "on-nosampled" in arms:
    band = re.search(r"\[spec-gate\] policy .*?LOW=(\d+) HIGH=(\d+)", log("on-gate", 1))
    if not band:
        fails.append("could not read the spec-gate band off the on-gate boot log — the "
                     "sampled verdict is policy-keyed and refuses to guess it")
        low = mark = None
    else:
        low, high = int(band.group(1)), int(band.group(2))
        # The sampled restore's watermark is SOLO, clamped by the band (worker.rs
        # `sampled_restore_watermark`). Derived here rather than hardcoded so the verdict
        # follows the policy the server actually booted with.
        mark = min(low, 1)
        print(f"server band: LOW={low} HIGH={high}; sampled-restore watermark (SOLO) {mark}; "
              f"measured concurrency c{conc}")
    gate_rs, nos_rs = med("on-gate", "req_s"), med("on-nosampled", "req_s")
    recovery = gate_rs / nos_rs if nos_rs else 0.0
    eng_gate = sum(summ("on-gate", p)["engaged"] for p in range(1, passes + 1))
    eng_nos = sum(summ("on-nosampled", p)["engaged"] for p in range(1, passes + 1))
    refusals = sum(len(re.findall(r"\[spec-restore-guard\].*REFUSED", log("on-gate", p)))
                   for p in range(1, passes + 1))
    print(f"SAMPLED c{conc} (temp {temp}): on-gate {gate_rs} req/s vs on-nosampled "
          f"{nos_rs} req/s = {recovery:.3f}x  [engaged rows: gate {eng_gate}, "
          f"nosampled {eng_nos}; guard refusals logged: {refusals}]")
    if "on-noguard" in arms:
        ng_rs = med("on-noguard", "req_s")
        eng_ng = sum(summ("on-noguard", p)["engaged"] for p in range(1, passes + 1))
        print(f"CONTROL on-noguard (MEMRA_SPEC_RESTORE_LOAD_GUARD=0): {ng_rs} req/s = "
              f"{ng_rs / off_rs:.3f}x off, {ng_rs / nos_rs:.3f}x on-nosampled, "
              f"engaged rows {eng_ng} — the un-guarded lever, reported not gated")
    if eng_nos != 0:
        fails.append(f"on-nosampled engaged spec on {eng_nos} rows — rollback door leaks")
    if mark is not None and conc > mark:
        # ABOVE the watermark: the guard must refuse, and refusing must buy back parity.
        if eng_gate != 0:
            fails.append(f"c{conc} > watermark {mark} but on-gate engaged spec on {eng_gate} "
                         "rows — the load guard did not hold above its own watermark")
        if refusals == 0:
            fails.append("no [spec-restore-guard] REFUSED line in any on-gate pass — a "
                         "guard that does not name itself is indistinguishable from a "
                         "mechanism that never ran (the v0.93.0 silent-refusal lesson)")
        if not (0.98 <= recovery <= 1.02):
            fails.append(f"c{conc} parity {recovery:.3f}x outside [0.98, 1.02] vs the "
                         "door-shut posture — refusing should cost and buy nothing here")
    elif mark is not None:
        # AT OR BELOW the watermark (SOLO): the tier the lever is FOR.
        if eng_gate == 0:
            fails.append(f"c{conc} <= watermark {mark} but sampled rows never engaged spec "
                         "in on-gate — the guard is refusing in the tier that pays")
        if recovery < 1.05:
            fails.append(f"sampled recovery {recovery:.3f}x < pre-registered 1.05x")
print("SPEC-CACHE GATE:", "FAIL — " + "; ".join(fails) if fails else "PASS")
sys.exit(1 if fails else 0)
PY
