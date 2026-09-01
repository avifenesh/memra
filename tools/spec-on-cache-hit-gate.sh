#!/usr/bin/env bash
# spec-on-cache-hit-gate.sh — spec must ENGAGE on prefix-cache hits (lane/spec-on-cache-hit,
# SAMPLED arm lane/sampled-hit-spec).
#
# The 2026-08-18 endpoint bench (darklanes ops/bench/endpoint-bench-20260818) caught
# production disengaging spec on every cache-hit row: repeated-prompt agent-loop traffic
# took the prefix-cache discount but decoded plain (~135 -> ~75 tok/s on qwen). This gate
# pins the fix on BOTH spec arms with the smallest local artifacts:
#
#   qwen arm (MTP head): cold spec publishes a draft-plane entry; the identical repeat
#     re-arms via the empty-suffix continuation; the extended repeat re-arms via the
#     suffix prime. usage.spec present with accepted > 0 on every hit row, AND the hit's
#     output bytes IDENTICAL to a spec-off boot serving the same hit (spec==plain law).
#   qwen SAMPLED cells (s*): the same two hit shapes at temperature > 0. WHY THEY EXIST:
#     v0.93.0 shipped the restore GREEDY-ONLY, and the DE deploy window then measured 3
#     cache hits / 3 plain downgrades / 0 restores — the paying traffic is sampled (the
#     OpenAI surface defaults to temperature 1.0), so the headline was inert in
#     production while every greedy gate stayed green. A gate that only asks greedy
#     questions cannot see that; these cells ask the sampled one.
#     ACCEPTANCE for sampled (the repo's sampled-spec contract, worker.rs step_session:
#     sampled spec is distributionally exact, NOT byte-equal to plain sampling — so
#     "spec == plain bytes" is unavailable and asking for it would be a fake gate):
#       (a) ENGAGEMENT — usage.spec.accepted > 0 on the hit, cached_tokens > 0;
#       (b) SEEDED IDENTITY vs the COLD spec leader — per seed, the full-cover hit's
#           bytes == the cold leader's bytes. Repeated over 3 seeds, that is an
#           equivalence-in-distribution proof against the path the hit replaces
#           (identical output for every seed => identical distribution over seeds);
#       (c) ACCEPTANCE PARITY — the full-cover hit's accepted/drafted == the cold
#           leader's exactly (the greedy arm's 0.460 == 0.460 property, under sampling);
#       (d) REPRODUCIBILITY of the suffix shape — two identical sampled suffix hits at
#           one seed are byte-equal (the run-spec sampled-gate precedent);
#       (e) REFUSAL IS REAL — a refusal must serve the hit PLAIN and SAY WHY on the
#           downgrade line. The cell is a PLANE-LESS entry (a greedy+penalized leader is
#           spec-ineligible upstream, so its entry carries no draft plane) and the sampled
#           repeat must refuse BY NAME. Until lane/sampled-spec-quality this cell was the
#           penalized-sampled refusal; that refusal was LIFTED when the burst's penalty
#           window learned to span the session, so the sp cells now assert the OPPOSITE —
#           a penalized sampled hit ENGAGES and reproduces its cold leader's bytes.
#   qwen BOUNDARY assertions (lane/sampled-spec-quality Item 1): the token a burst emits at
#     its OWN boundary was an ARGMAX in both regimes, so a sampled stream took a greedy
#     token once per burst AND every sampled request began with the same greedy token no
#     matter the seed. The spec-on boot runs with MEMRA_SPEC_BOUNDARY_TRACE=1; the gate
#     asserts all three boundary SITES fired and at least one draw DEVIATED from the argmax
#     it replaced. The customer-visible half is the one-token cells (bg/b<seed>,
#     max_tokens=1): pre-lane EVERY sampled request's first token was the greedy one, for
#     every seed; the gate requires at least one seed to differ from the greedy cell now, and
#     the teeth arm requires all of them to match it.
#   qwen GROWTH cells (g1/g2/g3, lane/sampled-spec-quality Item 3): a 3-turn growing
#     conversation. Turn 2 restores turn 1's entry AND republishes its own prompt-end
#     boundary, so turn 3 must hit a STRICTLY LONGER prefix than turn 2 did. Pre-lane,
#     publication was armed for COLD sessions only, so a namespace learned exactly one
#     boundary and the cached fraction decayed as the conversation grew.
#   gemma arm (assistant drafter over trunk KV): a sampled leader publishes a plain
#     entry; the greedy extended repeat re-arms gspec from the restored carrier; the
#     greedy full-cover repeat stays PLAIN by design (documented decline) — asserted.
#     Gemma has NO sampled cells on purpose: the gemma verify is a pure argmax walk, so
#     its whole spec route is greedy-only upstream of any cache question (worker.rs
#     gspec_k). Sampled gemma hits serving plain is the documented route, not this lane.
#
# usage:
#   spec-on-cache-hit-gate.sh qwen  <model.gguf>              <server_bin> <evidence_dir>
#   spec-on-cache-hit-gate.sh gemma <model.gguf> <draft.gguf> <server_bin> <evidence_dir>
#
# MEMRA_HITGATE_TEETH=1 flips every door this arc added into its ROLLBACK posture — the
# spec-on boot takes MEMRA_SPEC_RESTORE_SAMPLED=0 MEMRA_SPEC_SAMPLED_BOUNDARY=0
# MEMRA_SPEC_PEN_SESSION=0 MEMRA_SPEC_RESTORE_REPUBLISH=0, i.e. the v0.93.0 posture in full
# — and the gate then REQUIRES the pre-lane behaviour: sampled hits stay plain, penalized
# sampled hits stay plain with the window door named, turn 3 does NOT hit longer than turn
# 2, no boundary draw happens, and every sampled request opens with the greedy token. That
# arm proves the rollbacks work AND that these cells have teeth (they fail on a
# v0.93.0-shaped binary rather than passing vacuously). Greedy must stay green in BOTH
# postures — that is the guard rail on every change in this arc.
#
# Boots its own servers one arm at a time (flock ${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}).
# Exit 0 = every assertion held. Evidence: <evidence_dir>/{arm}-{on,off}-r{1,2,3}.json,
# <evidence_dir>/qwen-on-s*.json and logs.
set -euo pipefail
ARM=$1
case "$ARM" in
qwen)
    MODEL=$2
    DRAFT=""
    BIN=$3
    EV=$4
    ;;
gemma)
    MODEL=$2
    DRAFT=$3
    BIN=$4
    EV=$5
    ;;
*)
    echo "usage: $0 qwen|gemma ..." >&2
    exit 2
    ;;
esac
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_GATE_PORT:-18099}
HERE=$(cd "$(dirname "$0")" && pwd)
# Port occupancy guard (GATE-INTEGRITY-20260819 A-16, deferred to this file's merge because
# the sampled lanes had it open). The curl probe inside boot() is a RESPONDER check and only
# sees a squatter that speaks this API; memra_port_guard is the occupancy check and fails
# closed when neither ss nor lsof can observe the port. Pre-flight only — $SERVER_PID is the
# flock wrapper's pid, so memra_port_owned would manufacture a false red.
[ -f "$HERE/port-guard.sh" ] || {
    echo "spec-on-cache-hit-gate: FAIL — $HERE/port-guard.sh is missing; refusing to bind unguarded" >&2
    exit 1
}
. "$HERE/port-guard.sh"
# EXTERNAL MTP DRAFTER for the qwen arm (2026-08-19). The qwen arm otherwise drafts off the
# trunk's OWN embedded head, which is a real config but is NOT the production one for an
# artifact whose drafter ships separately. Set MEMRA_GATE_MTP_DRAFT=<draft.gguf> to attach the
# real drafter — and note the seam: the qwen attach is MEMRA_MTP_DRAFT, NOT MEMRA_DRAFT (the
# gemma assistant-drafter flag, which on a qwen model attaches NOTHING while still flipping
# wkv_on()/fa_f16pv_on()/the MMQ-SK form). docs/FLAGS.md 'SEAM TRAP'. When set, every boot in
# this arm ASSERTS the attach line: a silent no-drafter run FAILS instead of passing on the
# embedded head.
MTP_DRAFT=${MEMRA_GATE_MTP_DRAFT:-}
mkdir -p "$EV"
SERVER_PID=""
DRAFT_FAILS=0
# Assert the SPEC-ON boot really loaded the drafter it was handed. Called only on spec-on
# boots: the gemma spec-off twin deliberately boots MEMRA_GEMMA4_SPEC=0, which does not load
# the assistant drafter at all, so asserting there would fail for the right config.
assert_mtp_drafter() { # $1 spec-on server log
    [ -n "$MTP_DRAFT" ] || return 0
    if ! "$HERE/assert-drafter-attached.sh" "$1" "$MTP_DRAFT"; then
        DRAFT_FAILS=$((DRAFT_FAILS + 1))
    fi
}
assert_gemma_drafter() { # $1 spec-on server log
    if ! "$HERE/assert-drafter-attached.sh" --gemma "$1" "$DRAFT"; then
        DRAFT_FAILS=$((DRAFT_FAILS + 1))
    fi
}

boot() { # $1 extra-env-string  $2 log
    memra_port_guard spec-on-cache-hit-gate "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving — refusing to boot over it"
        return 1
    fi
    # shellcheck disable=SC2086
    # PREFIX BUDGET PINNED, not derived (lane/sampled-hit-spec): the sampled cells use one
    # PC-ISO namespace each, and the derived budget on this rig came to 349MB with the run
    # reaching 327MB resident — a card with less free VRAM would evict a cell's entry and
    # turn its "hit" row into a miss, i.e. a FAIL that says nothing about the code.
    local mtp=()
    [ -n "$MTP_DRAFT" ] && mtp=("MEMRA_MTP_DRAFT=$MTP_DRAFT")
    flock -w 300 "$GPU_LOCK" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
        "MEMRA_PREFIX_CACHE_MB=${MEMRA_HITGATE_CACHE_MB:-2048}" \
        "${mtp[@]}" $1 "$BIN" >"$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        if curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || {
            echo "server died during boot:"
            tail -20 "$2"
            return 1
        }
        sleep 2
    done
    echo "server never became ready"
    return 1
}
stop() {
    # kill the SERVER, not the flock wrapper (the spec-cache-gate.sh lesson: killing the
    # wrapper orphans the server and the next boot silently reuses it on the same port).
    pkill -x memra-server 2>/dev/null || true
    for _ in $(seq 1 30); do
        curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || {
            SERVER_PID=""
            sleep 2
            return 0
        }
        sleep 1
    done
    pkill -9 -x memra-server 2>/dev/null || true
    SERVER_PID=""
    sleep 3
}
trap stop EXIT

# Long deterministic prompt (> PREFIX_CACHE_MIN_TOKENS=64 tokens for every tokenizer here)
# + a short extension for the suffix-hit cell.
PROMPT="You are cataloguing the maintenance history of a small observatory. List, in order, \
the twelve monthly inspection tasks for the primary mirror cell, the guidance system, the \
dome rotation drive, and the weather station mast, and for each task name the tool required, \
the expected duration in minutes, the sign-off role, and the failure symptom that would \
trigger an early re-inspection. Be systematic and terse. Begin with January and do not skip \
any month. After the twelve months, add a short paragraph on annual recalibration."
EXT=" Finally, state which single month carries the highest workload and why."
# Second extension for the GROWTH cells: turn 3 of a growing conversation.
EXT2=" Also note the two tasks most often deferred and the risk of deferring them."

req() { # $1 prompt $2 temp $3 out-json [$4 seed] [$5 cache_salt] [$6 freq_pen] [$7 max_tokens]
    python3 - "$PORT" "$1" "$2" "$3" "${4:-7}" "${5:-}" "${6:-0}" "${7:-48}" <<'PY'
import json, sys, urllib.request
port, prompt, temp, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
seed, salt, freq, maxtok = int(sys.argv[5]), sys.argv[6], float(sys.argv[7]), int(sys.argv[8])
body = {"model": "gate", "prompt": prompt, "max_tokens": maxtok, "temperature": temp}
if temp > 0:
    # An OMITTED seed is fresh entropy per request (main.rs sampler_config), so every
    # sampled cell pins one explicitly — otherwise the identity cells would compare two
    # different streams and "fail" for the wrong reason.
    body["seed"] = seed
if salt:
    body["cache_salt"] = salt  # PC-ISO namespace: one cell's entries stay its own
if freq:
    body["frequency_penalty"] = freq
r = urllib.request.urlopen(
    urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    ),
    timeout=600,
)
resp = json.load(r)
json.dump(resp, open(out, "w"), indent=1)
PY
}

FAILS=0
check() { # $1 name  $2 python-bool-expr over r1/r2/r3 (loaded json)
    local name=$1 expr=$2 arm=$3
    if python3 - "$EV" "$arm" "$expr" <<'PY'
import json, sys
ev, arm, expr = sys.argv[1], sys.argv[2], sys.argv[3]
def load(n):
    try:
        return json.load(open(f"{ev}/{arm}-r{n}.json"))
    except FileNotFoundError:
        return None
r1, r2, r3 = load(1), load(2), load(3)
def text(r):
    return r["choices"][0]["text"]
def cached(r):
    return r["usage"]["prompt_tokens_details"]["cached_tokens"]
def spec(r):
    return r["usage"].get("spec")
sys.exit(0 if eval(expr) else 1)
PY
    then
        echo "  ok: $name"
    else
        echo "  FAIL: $name"
        FAILS=$((FAILS + 1))
    fi
}

TEETH=${MEMRA_HITGATE_TEETH:-0}
SAMPLED_TEMP=${MEMRA_HITGATE_TEMP:-0.8}
SAMPLED_SEEDS=${MEMRA_HITGATE_SEEDS:-"7 1234 99991"}
# ---- BOUNDARY-PROBE TEMPERATURE: 4.0, and here is the arithmetic that fixes it there. ----
#
# The probe asserts "at least one of K=3 seeds draws a FIRST token different from the greedy
# argmax". Its power is therefore  1 - (1-p)^K  where p = P(one seed deviates) on THIS fixture
# and THIS artifact, and its false-fail rate is  (1-p)^K.  Both are properties of the fixture,
# not of the code — which is what makes an underpowered default worse than no gate at all.
#
# MEASURED on the 27B production artifact (Qwen3.8-27B-NVFP4-Q5K-mtp, sha 1facf36c2db359dc),
# 12 seeds per rung, `tools/btemp-power.sh` (banked; re-run it on any new artifact):
#
#   | BTEMP | deviating/12 |    p̂   | 95% CP lower bound on p | P(false fail at K=3) | distinct |
#   |-------|--------------|--------|-------------------------|----------------------|----------|
#   | 1.5   |     1/12     | 0.083  |          0.004          |  0.771  ->  77% (!)  |    2     |
#   | 2.5   |     9/12     | 0.750  |          0.472          |  0.016                |   10     |
#   | 4.0   |    12/12     | 1.000  |          0.779          |  <= 0.011 (<= 1.1%)   |   12     |
#   | 8.0   |    12/12     | 1.000  |          0.779          |  <= 0.011             |   12     |
#
#   p̂ = deviating/12.  The bound for a 12/12 rung is Clopper-Pearson one-sided:
#   p >= alpha^(1/n) = 0.05^(1/12) = 0.779, so P(false fail) = (1-p)^3 <= 0.221^3 = 0.0108.
#   For the 1/12 rung the point estimate is what matters: (1 - 1/12)^3 = 0.771.
#
# So the SHIPPED 1.5 DEFAULT FAILED THIS ASSERTION 77% OF THE TIMES IT RAN on the 27B — a gate
# that cries wolf gets ignored, which is worse than no gate. Raised to 4.0 on 2026-08-19.
# Truncation was NOT the mechanism (memra's omitted filters default to top_p 1.0 / top_k 0,
# both disabled): the position after this fixture's prompt is a structural paragraph break and
# the mass simply is not there at 1.5.
#
# WHY TEMPERATURE AND NOT A NEW FIXTURE. Moving the fixture to a high-entropy position would
# also work, but its power would be a property of the artifact's own distribution at that
# position — i.e. it would need re-measuring for every model this gate is ever pointed at. A
# temperature of 4.0 flattens whatever distribution is there, so the probe's power travels with
# the tool. The teeth arm gets the same treatment for free and is STRONGER for it: at 4.0 a
# live sampler deviates on every seed, so doors-shut returning the argmax is a real assertion
# instead of one that passes because nothing was drawing.
#
# The cross-check below (`BDEV`) makes a future underpowered fixture SELF-DIAGNOSING rather
# than silent: if this probe fails while the independent traced-draw instrument shows draws
# deviating, the gate says FIXTURE, not code.
BOUNDARY_TEMP=${MEMRA_HITGATE_BTEMP:-4.0}

# All sampled-cell assertions in one pass (12+ separate check() calls would be unreadable,
# and every cell needs cross-file comparisons the r1/r2/r3 helper cannot express).
sampled_checks() {
    rm -f "$EV/BTEMP-PROBE-FAILED"
    python3 - "$EV" "$TEETH" "$SAMPLED_SEEDS" "$BOUNDARY_TEMP" <<'PY'
import json, os, sys
ev, teeth, seeds = sys.argv[1], sys.argv[2] == "1", sys.argv[3].split()
btemp = sys.argv[4]
fails = 0


def load(name):
    with open(f"{ev}/{name}.json") as f:
        return json.load(f)


def text(r):
    return r["choices"][0]["text"]


def cached(r):
    return r["usage"]["prompt_tokens_details"]["cached_tokens"]


def spec(r):
    return r["usage"].get("spec")


def ok(name, cond):
    global fails
    print(f"  {'ok' if cond else 'FAIL'}: {name}")
    if not cond:
        fails += 1


for s in seeds:
    cold, hit = load(f"qwen-on-s{s}-cold"), load(f"qwen-on-s{s}-hit")
    ok(f"s{s} sampled cold engages spec, zero cached",
       spec(cold) is not None and spec(cold)["drafted"] > 0 and cached(cold) == 0)
    ok(f"s{s} sampled hit is a FULL-COVER cache hit",
       cached(hit) > 0 and cached(hit) == hit["usage"]["prompt_tokens"])
    if teeth:
        # ROLLBACK posture: the door must hold sampled hits on the plain path.
        ok(f"s{s} sampled hit stays PLAIN under MEMRA_SPEC_RESTORE_SAMPLED=0",
           spec(hit) is None)
    else:
        ok(f"s{s} sampled hit SPEC ENGAGED (accepted > 0)",
           spec(hit) is not None and spec(hit)["accepted"] > 0)
        # (b) seeded identity vs the cold leader — the sampled arm's exactness standard.
        ok(f"s{s} sampled hit bytes == cold leader bytes (same seed)", text(hit) == text(cold))
        # (c) acceptance parity: the restored draft plane is the cold plane, exactly.
        ok(f"s{s} sampled hit acceptance == cold acceptance exactly",
           spec(hit) is not None
           and (spec(hit)["accepted"], spec(hit)["drafted"])
           == (spec(cold)["accepted"], spec(cold)["drafted"]))

lead, x1, x2 = load("qwen-on-sx-lead"), load("qwen-on-sx1"), load("qwen-on-sx2")
ok("sx sampled leader engages spec, zero cached",
   spec(lead) is not None and cached(lead) == 0)
ok("sx sampled SUFFIX hit restores a strict prefix",
   0 < cached(x1) < x1["usage"]["prompt_tokens"])
if teeth:
    ok("sx sampled suffix hit stays PLAIN under the rollback door", spec(x1) is None)
else:
    ok("sx sampled SUFFIX hit SPEC ENGAGED (accepted > 0)",
       spec(x1) is not None and spec(x1)["accepted"] > 0)
# (d) reproducibility holds in both postures: same seed, same shape, same bytes.
ok("sx sampled suffix hit reproduces byte-for-byte at one seed", text(x1) == text(x2))

# --- PENALIZED SAMPLED hits (lane/sampled-spec-quality Item 2 lifted the v2 refusal) ---
# The refusal existed because the burst seeded its penalty window from the `prompt` slice it
# was handed, and a converted hit hands it none — so a penalized restore would have decoded
# against a window the client never asked for. The window now spans the session's committed
# tokens, and a restored session's committed IS the whole prompt, so the restore reproduces
# the cold leader EXACTLY like the unpenalized cells do. Under the teeth posture
# (MEMRA_SPEC_PEN_SESSION=0) the burst-local window is back and the refusal must return.
plead, phit = load("qwen-on-sp-lead"), load("qwen-on-sp-hit")
ok("sp penalized sampled leader engages spec (entry has a draft plane)",
   spec(plead) is not None and cached(plead) == 0)
if teeth:
    ok("sp penalized sampled hit stays PLAIN under the burst-local window door",
       spec(phit) is None and cached(phit) > 0)
else:
    ok("sp penalized sampled hit is a FULL-COVER hit",
       cached(phit) > 0 and cached(phit) == phit["usage"]["prompt_tokens"])
    ok("sp penalized sampled hit SPEC ENGAGED (accepted > 0)",
       spec(phit) is not None and spec(phit)["accepted"] > 0)
    ok("sp penalized sampled hit bytes == cold leader bytes (same seed)",
       text(phit) == text(plead))
    ok("sp penalized sampled hit acceptance == cold acceptance exactly",
       spec(phit) is not None
       and (spec(phit)["accepted"], spec(phit)["drafted"])
       == (spec(plead)["accepted"], spec(plead)["drafted"]))

# (e) A REFUSAL IS STILL REAL, in both postures, and it names itself: the `np` cell's leader
# is GREEDY+penalized, which is spec-INELIGIBLE upstream (`greedy_penalized`), so it serves
# plain and publishes an entry with NO draft plane. The sampled repeat must refuse — this is
# the cell that keeps "refuse rather than guess" observable now that the penalty refusal is
# gone, and it is the exact shape of the v0.93.0 production downgrade.
nlead, nhit = load("qwen-on-np-lead"), load("qwen-on-np-hit")
ok("np greedy+penalized leader serves PLAIN (publishes a plane-less entry)",
   spec(nlead) is None and cached(nlead) == 0)
ok("np sampled hit on a plane-less entry stays PLAIN (refusal is real)",
   spec(nhit) is None and cached(nhit) > 0)

# --- ITEM 1, the CUSTOMER-VISIBLE probe. Pre-lane every sampled request's FIRST token was
# `argmax(prime_logits)` — the same token the greedy request emits from the same row, for
# every seed and every temperature. These cells are `max_tokens=1`, so the response text IS
# exactly that one token and nothing downstream can muddy the comparison (an earlier draft
# compared the first 4 characters of 48-token responses and was NOT decisive: the first token
# was "\n\n" in every cell and the visible difference came from token 2). Deterministic, not
# flaky: fixed prompt + fixed seeds. The cell runs at a HIGH temperature on purpose so the
# draw is unlikely to coincide with the argmax — and that "on purpose" is now MEASURED rather
# than asserted: the in-file comment used to claim 1.5 was hot enough "on any artifact,
# including the 27B production one", and the 96GB window measured p(deviate) = 1/12 there,
# i.e. a 77% false-fail rate. Default raised to 4.0 (12/12); see the BOUNDARY_TEMP block.
bg = load("qwen-on-bg")
bs = [load(f"qwen-on-b{s}") for s in seeds]
if teeth:
    ok("teeth: every sampled FIRST token is the greedy argmax (the pre-lane defect)",
       all(text(b) == text(bg) for b in bs))
else:
    drawn = any(text(b) != text(bg) for b in bs)
    ok("sampled FIRST token is drawn, not the greedy argmax (at least one seed differs)",
       drawn)
    print("     first tokens: greedy=%r sampled=%r  (BTEMP=%s, %d distinct over %d seeds)"
          % (text(bg), [text(b) for b in bs], btemp,
             len({text(b) for b in bs}), len(bs)))
    if not drawn:
        # FIXTURE-vs-CODE handoff. This probe's power is a property of the fixture AT THIS
        # TEMPERATURE, so a 0-of-K result is ambiguous on its own. The shell reads this marker
        # against the INDEPENDENT traced-draw instrument (`[spec-boundary] deviates=1`) and
        # names the class, so an underpowered fixture can never again be reported as a code
        # failure — the failure mode that made this red worth ignoring.
        with open(f"{ev}/BTEMP-PROBE-FAILED", "w") as f:
            f.write("%s %d %d\n" % (btemp, len(bs), len({text(b) for b in bs})))
sys.exit(1 if fails else 0)
PY
}

# GROWTH cells (Item 3): a 3-turn growing conversation must learn a LONGER boundary each turn.
growth_checks() {
    python3 - "$EV" "$TEETH" <<'PY'
import json, sys
ev, teeth = sys.argv[1], sys.argv[2] == "1"
fails = 0


def load(name):
    with open(f"{ev}/{name}.json") as f:
        return json.load(f)


def text(r):
    return r["choices"][0]["text"]


def cached(r):
    return r["usage"]["prompt_tokens_details"]["cached_tokens"]


def prompt_toks(r):
    return r["usage"]["prompt_tokens"]


def spec(r):
    return r["usage"].get("spec")


def ok(name, cond):
    global fails
    print(f"  {'ok' if cond else 'FAIL'}: {name}")
    if not cond:
        fails += 1


g1, g2, g3, g4 = (load(f"qwen-on-g{i}") for i in (1, 2, 3, 4))
ok("g1 turn 1 is cold and engages spec",
   cached(g1) == 0 and spec(g1) is not None and spec(g1)["drafted"] > 0)
ok("g2 turn 2 restores turn 1's boundary (suffix-fed hit)",
   0 < cached(g2) < prompt_toks(g2) and spec(g2) is not None and spec(g2)["accepted"] > 0)
ok("g3 turn 3 engages spec", spec(g3) is not None and spec(g3)["accepted"] > 0)
if teeth:
    # Pre-lane: publication was cold-session-only, so turn 3 re-hits TURN 1's boundary and
    # the cached fraction stops growing. That is finding (d), reproduced on demand.
    ok("teeth: turn 3 hits the SAME boundary as turn 2 (no extended entry published)",
       cached(g3) == cached(g2))
    ok("teeth: the turn-2 repeat is not a full-cover hit (nothing was republished)",
       cached(g4) == cached(g2))
else:
    # THE Item 3 assertion.
    ok("g3 turn 3 hits a STRICTLY LONGER prefix than turn 2 did",
       cached(g3) > cached(g2))
    # ENTRY ACCOUNTING: the republished boundary is turn 2's own prompt END — whole-entry
    # semantics, never mid-entry (the rolled-back partial-restore hazard stays closed).
    ok("g3's restored prefix == turn 2's whole prompt (whole-entry boundary)",
       cached(g3) == prompt_toks(g2))
    # STATE CORRECTNESS, and why it is THIS comparison. A republished entry is a snapshot of
    # g2's own live boundary state, so restoring it must reproduce g2's own continuation
    # byte-for-byte: both sides continue from the same boundary through the same program, so a
    # difference can only be a lossy snapshot/restore (wrong GDN state, wrong boundary hidden,
    # wrong logits row). The spec-off twin is NOT the reference here — the plain path cannot
    # publish an extended entry at all, so it restores a SHORTER boundary and primes a longer
    # suffix; comparing against it measures prefill segmentation (the banked r3 two-programs
    # class), not this mechanism. See SAMPLED-QUALITY.md for the measured consequence.
    ok("g4 (full-cover hit on the REPUBLISHED entry) is a full-cover hit",
       cached(g4) == prompt_toks(g4) and cached(g4) == prompt_toks(g2))
    ok("g4 SPEC ENGAGED on the republished entry",
       spec(g4) is not None and spec(g4)["accepted"] > 0)
    ok("g4 reproduces its publisher's continuation byte-for-byte (snapshot round-trip)",
       text(g4) == text(g2))
    ok("g4 acceptance == its publisher's acceptance exactly",
       spec(g4) is not None and spec(g2) is not None
       and (spec(g4)["accepted"], spec(g4)["drafted"])
       == (spec(g2)["accepted"], spec(g2)["drafted"]))
sys.exit(1 if fails else 0)
PY
}

if [ "$ARM" = qwen ]; then
    # MEMRA_SPEC_BOUNDARY_TRACE=1 is diagnostics-only (one stderr line per boundary draw) and
    # is what makes Item 1's assertions — and the MEASURED boundary rate — observable instead
    # of estimated. It is on in BOTH postures: the teeth arm asserts the lines are ABSENT.
    if [ "$TEETH" = 1 ]; then
        echo "== qwen arm: spec-on boot, TEETH/ROLLBACK posture (every door shut) =="
        boot "MEMRA_SPEC_BOUNDARY_TRACE=1 MEMRA_SPEC_RESTORE_SAMPLED=0 \
              MEMRA_SPEC_SAMPLED_BOUNDARY=0 MEMRA_SPEC_PEN_SESSION=0 \
              MEMRA_SPEC_RESTORE_REPUBLISH=0" "$EV/qwen-on-server.log"
        assert_mtp_drafter "$EV/qwen-on-server.log"
    else
        echo "== qwen arm: spec-on boot =="
        boot "MEMRA_SPEC_BOUNDARY_TRACE=1" "$EV/qwen-on-server.log"
        assert_mtp_drafter "$EV/qwen-on-server.log"
    fi
    req "$PROMPT" 0 "$EV/qwen-on-r1.json"      # cold: spec engages, publishes seed entry
    req "$PROMPT" 0 "$EV/qwen-on-r2.json"      # identical repeat: full-cover continuation
    req "$PROMPT$EXT" 0 "$EV/qwen-on-r3.json"  # extended repeat: suffix-prime restore
    # --- SAMPLED cells, each in its own PC-ISO namespace so the greedy rows above and every
    # other cell keep their own entries (a shared namespace would make cell 2's "cold"
    # leader a hit on cell 1's entry).
    for SEED in $SAMPLED_SEEDS; do
        req "$PROMPT" "$SAMPLED_TEMP" "$EV/qwen-on-s$SEED-cold.json" "$SEED" "samp-$SEED"
        req "$PROMPT" "$SAMPLED_TEMP" "$EV/qwen-on-s$SEED-hit.json" "$SEED" "samp-$SEED"
    done
    req "$PROMPT" "$SAMPLED_TEMP" "$EV/qwen-on-sx-lead.json" 7 samp-x
    req "$PROMPT$EXT" "$SAMPLED_TEMP" "$EV/qwen-on-sx1.json" 7 samp-x
    req "$PROMPT$EXT" "$SAMPLED_TEMP" "$EV/qwen-on-sx2.json" 7 samp-x
    req "$PROMPT" "$SAMPLED_TEMP" "$EV/qwen-on-sp-lead.json" 7 samp-pen 0.5
    req "$PROMPT" "$SAMPLED_TEMP" "$EV/qwen-on-sp-hit.json" 7 samp-pen 0.5
    # np cell: a GREEDY+penalized leader is spec-ineligible upstream, so its entry carries no
    # draft plane and the sampled repeat must refuse BY NAME (the live refusal cell).
    req "$PROMPT" 0 "$EV/qwen-on-np-lead.json" 7 samp-noplane 0.5
    req "$PROMPT" "$SAMPLED_TEMP" "$EV/qwen-on-np-hit.json" 7 samp-noplane
    # ONE-TOKEN boundary cells (Item 1's customer-visible probe): the response text IS the
    # boundary token. Own namespace per cell so none of them can hit another's entry.
    req "$PROMPT" 0 "$EV/qwen-on-bg.json" 7 bnd-g 0 1
    for SEED in $SAMPLED_SEEDS; do
        req "$PROMPT" "$BOUNDARY_TEMP" "$EV/qwen-on-b$SEED.json" "$SEED" "bnd-$SEED" 0 1
    done
    # GROWTH cells (Item 3): three turns of one conversation, greedy so byte identity against
    # the spec-off twin is available on the turn that hits the REPUBLISHED entry.
    req "$PROMPT" 0 "$EV/qwen-on-g1.json" 7 grow
    req "$PROMPT$EXT" 0 "$EV/qwen-on-g2.json" 7 grow
    req "$PROMPT$EXT$EXT2" 0 "$EV/qwen-on-g3.json" 7 grow
    # g4 repeats turn 2 EXACTLY, so it is a full-cover hit on the entry turn 2 REPUBLISHED.
    # This is the state-correctness oracle for Item 3 (see growth_checks): a republished entry
    # is a snapshot of g2's own live boundary state, so restoring it must reproduce g2's own
    # continuation byte-for-byte. Same program on both sides — no prefill-segmentation confound.
    req "$PROMPT$EXT" 0 "$EV/qwen-on-g4.json" 7 grow
    stop
    if grep -q "\[prefix-cache\] spec restore:" "$EV/qwen-on-server.log"; then
        echo "  ok: server log shows spec restore"
    else
        echo "  FAIL: no '[prefix-cache] spec restore:' in server log"
        FAILS=$((FAILS + 1))
    fi
    check "r1 cold has spec + zero cached" \
        "spec(r1) is not None and spec(r1)['drafted'] > 0 and cached(r1) == 0" qwen-on
    check "r2 hit has cached tokens" "cached(r2) > 0" qwen-on
    check "r2 hit SPEC ENGAGED (accepted > 0)" \
        "spec(r2) is not None and spec(r2)['accepted'] > 0" qwen-on
    check "r3 extended hit has cached tokens" "cached(r3) > 0" qwen-on
    check "r3 extended hit SPEC ENGAGED (accepted > 0)" \
        "spec(r3) is not None and spec(r3)['accepted'] > 0" qwen-on
    check "r2 text == r1 text (deterministic repeat)" "text(r2) == text(r1)" qwen-on

    echo "-- sampled cells (temperature $SAMPLED_TEMP, seeds: $SAMPLED_SEEDS, teeth=$TEETH) --"
    sampled_checks || FAILS=$((FAILS + 1))
    echo "-- growth cells (Item 3: a growing conversation must learn a longer boundary) --"
    growth_checks || FAILS=$((FAILS + 1))

    # --- Item 1 boundary-draw log assertions (see the header) ---
    echo "-- boundary draws (Item 1) --"
    BLINES=$(grep -c "\[spec-boundary\] site=" "$EV/qwen-on-server.log" || true)
    BDEV=$(grep -c "\[spec-boundary\].*deviates=1" "$EV/qwen-on-server.log" || true)
    if [ "$TEETH" = 1 ]; then
        # the door is shut: not a single boundary token may be drawn.
        if [ "$BLINES" = 0 ]; then
            echo "  ok: teeth: no boundary draw happened (MEMRA_SPEC_SAMPLED_BOUNDARY=0)"
        else
            echo "  FAIL: teeth: $BLINES boundary draws with the door shut"
            FAILS=$((FAILS + 1))
        fi
    else
        # Every boundary SITE must be exercised by this cell set, or a site could regress
        # silently: the cold prime's first token, a continuation burst's stashed token
        # (max_tokens 48 > MEMRA_SPEC_BURST 32, so every sampled cell crosses one boundary),
        # and a converted full-cover hit's seed.
        for SITE in cold-prime burst-tail-commit restore-full-cover; do
            if grep -q "\[spec-boundary\] site=$SITE " "$EV/qwen-on-server.log"; then
                echo "  ok: boundary site $SITE fired"
            else
                echo "  FAIL: boundary site $SITE never fired (untested code path)"
                FAILS=$((FAILS + 1))
            fi
        done
        # ... and at least one draw must have DEVIATED from the argmax it replaced, otherwise
        # the fix is indistinguishable from the defect on this cell set.
        if [ "$BDEV" -gt 0 ]; then
            echo "  ok: $BDEV of $BLINES boundary draws deviated from the pre-lane argmax"
        else
            echo "  FAIL: every boundary draw returned the argmax (fix not demonstrated)"
            FAILS=$((FAILS + 1))
        fi
        # FIXTURE-vs-CODE attribution for the one-token probe (2026-08-19). Two independent
        # instruments read the same property: the K-seed one-token probe (fixture-sensitive)
        # and the traced boundary draws (fixture-insensitive — every sampled cell crosses a
        # burst boundary). When they disagree, SAY WHICH, instead of leaving the operator to
        # re-derive it from a temperature ladder as the 96GB window had to.
        if [ -f "$EV/BTEMP-PROBE-FAILED" ]; then
            read -r PT PN PD < "$EV/BTEMP-PROBE-FAILED"
            if [ "$BDEV" -gt 0 ]; then
                echo "  ATTRIBUTION: the one-token probe FAILED but $BDEV of $BLINES traced"
                echo "    draws DEVIATED in the same run — the sampler IS drawing, so this is"
                echo "    an UNDERPOWERED FIXTURE at MEMRA_HITGATE_BTEMP=$PT ($PD distinct"
                echo "    token(s) over $PN seeds), NOT a code failure. Measure the probe's"
                echo "    power on this artifact with tools/btemp-power.sh and raise BTEMP"
                echo "    (or move the fixture) until p(deviate) makes (1-p)^K negligible."
            else
                echo "  ATTRIBUTION: the one-token probe failed AND zero traced draws deviated"
                echo "    — both instruments agree, so this is a CODE failure, not the fixture."
            fi
        fi
    fi
    # The refusal must SAY WHY on the downgrade line — v0.93.0's silent refusal is how the
    # greedy-only scope survived a deploy verification (0 restores AND 0 declines).
    # Posture-scoped on purpose: with the rollback door shut, the door refusal short-circuits
    # every sampled hit (worker.rs spec_restore_refusal checks it before the penalty rule), so
    # the penalty reason cannot appear. The first teeth run asserted it unconditionally and
    # failed for exactly that reason — the teeth arm caught a gate defect on its first outing.
    # The live refusal cell in the DEFAULT posture is the plane-less entry (the penalized
    # refusal was lifted — see the header). It must name itself.
    if grep -q "reason: entry carries no draft plane" "$EV/qwen-on-server.log"; then
        echo "  ok: plane-less refusal names itself in the log"
    else
        echo "  FAIL: no named plane-less refusal in the server log"
        FAILS=$((FAILS + 1))
    fi
    if [ "$TEETH" = 1 ]; then
        # BOTH doors must name themselves in the one teeth log. The refusal order was chosen
        # for exactly this (the window reason is reported before the fleet door), because a
        # short-circuit is how the first teeth run produced a green-looking half-tested arm.
        if grep -q "reason: sampled restore disabled" "$EV/qwen-on-server.log"; then
            echo "  ok: sampled-restore door names itself in the log"
        else
            echo "  FAIL: MEMRA_SPEC_RESTORE_SAMPLED=0 did not name itself in the log"
            FAILS=$((FAILS + 1))
        fi
        if grep -q "reason: sampled request with an active penalty window and a burst-local" \
            "$EV/qwen-on-server.log"; then
            echo "  ok: burst-local-window door names itself in the log"
        else
            echo "  FAIL: MEMRA_SPEC_PEN_SESSION=0 did not name itself in the log"
            FAILS=$((FAILS + 1))
        fi
    fi

    echo "== qwen arm: spec-off twin boot (identity reference) =="
    boot "MEMRA_SERVE_SPEC=0" "$EV/qwen-off-server.log"
    req "$PROMPT" 0 "$EV/qwen-off-r1.json"
    req "$PROMPT" 0 "$EV/qwen-off-r2.json"
    req "$PROMPT$EXT" 0 "$EV/qwen-off-r3.json"
    # growth turns 1-2 on the plain path: same boundaries, same programs, so spec==plain
    # applies. Turn 3 is deliberately NOT compared — with extended-entry publication on, the
    # spec boot restores a boundary the plain path CANNOT publish (the plain tier refuses a
    # mid-entry restore, so it re-primes a longer suffix), and a cross-segmentation diff
    # measures the banked r3 two-programs class instead of this lane. Turn 3's own oracle is
    # the g4 round-trip cell above; the measured cross-segmentation delta is banked as a
    # finding in darklanes research/spec-cache-20260818/SAMPLED-QUALITY.md.
    req "$PROMPT" 0 "$EV/qwen-off-g1.json" 7 grow
    req "$PROMPT$EXT" 0 "$EV/qwen-off-g2.json" 7 grow
    stop
    check "off-boot rows carry no spec" \
        "spec(r1) is None and spec(r2) is None and spec(r3) is None" qwen-off
    for n in r1 r2 r3 g1 g2; do
        if python3 -c "
import json,sys
a=json.load(open('$EV/qwen-on-$n.json'));b=json.load(open('$EV/qwen-off-$n.json'))
sys.exit(0 if a['choices'][0]['text']==b['choices'][0]['text'] else 1)"; then
            echo "  ok: $n spec==plain byte identity"
        else
            echo "  FAIL: $n spec-on text != spec-off text (identity law)"
            FAILS=$((FAILS + 1))
        fi
    done
else
    echo "== gemma arm: spec-on boot (drafter attached) =="
    boot "MEMRA_DRAFT=$DRAFT" "$EV/gemma-on-server.log"
    assert_gemma_drafter "$EV/gemma-on-server.log"
    req "$PROMPT" 0.7 "$EV/gemma-on-r1.json"   # sampled leader: plain path, publishes seed
    req "$PROMPT$EXT" 0 "$EV/gemma-on-r2.json" # greedy extended: gspec restored carrier
    req "$PROMPT" 0 "$EV/gemma-on-r3.json"     # greedy full-cover: PLAIN by design
    stop
    check "r1 sampled leader is plain + cold" "spec(r1) is None and cached(r1) == 0" gemma-on
    check "r2 hit has cached tokens" "cached(r2) > 0" gemma-on
    check "r2 hit SPEC ENGAGED (accepted > 0)" \
        "spec(r2) is not None and spec(r2)['accepted'] > 0" gemma-on
    check "r3 full-cover hit stays PLAIN (documented decline)" \
        "spec(r3) is None and cached(r3) > 0" gemma-on

    echo "== gemma arm: spec-off twin boot (identity reference) =="
    boot "MEMRA_SERVE_SPEC=0 MEMRA_GEMMA4_SPEC=0 MEMRA_DRAFT=$DRAFT" "$EV/gemma-off-server.log"
    req "$PROMPT" 0.7 "$EV/gemma-off-r1.json"
    req "$PROMPT$EXT" 0 "$EV/gemma-off-r2.json"
    req "$PROMPT" 0 "$EV/gemma-off-r3.json"
    stop
    for n in 2 3; do
        if python3 -c "
import json,sys
a=json.load(open('$EV/gemma-on-r$n.json'));b=json.load(open('$EV/gemma-off-r$n.json'))
sys.exit(0 if a['choices'][0]['text']==b['choices'][0]['text'] else 1)"; then
            echo "  ok: r$n spec==plain byte identity"
        else
            echo "  FAIL: r$n spec-on text != spec-off text (identity law)"
            FAILS=$((FAILS + 1))
        fi
    done
fi

# A silent no-drafter run must FAIL, not pass. Counted into the verdict separately so the
# line says which class it was: an attach failure means the run measured the wrong config,
# and none of the assertions above were testing what they claimed to test.
if [ "$DRAFT_FAILS" -gt 0 ]; then
    echo "  FAIL: $DRAFT_FAILS drafter-attach assertion(s) — the drafter handed to this gate"
    echo "        never loaded, so this run served on the trunk's own head"
    FAILS=$((FAILS + DRAFT_FAILS))
fi
if [ "$FAILS" -eq 0 ]; then
    echo "SPEC-ON-CACHE-HIT GATE: ALL GREEN ($ARM)"
else
    echo "SPEC-ON-CACHE-HIT GATE: $FAILS FAILURE(S) ($ARM)"
    exit 1
fi
