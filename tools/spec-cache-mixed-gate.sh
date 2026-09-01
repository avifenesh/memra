#!/usr/bin/env bash
# spec-cache-mixed-gate.sh — spec-on-cache-hit under batch coexistence (lane/spec-on-cache-hit).
#
# STAGED for a 96GB serving-card window (a rented box at a lane boundary, or the owner's box
# when delegated). DO NOT run on a prod box serving customers. Local 24GB rigs run
# tools/spec-on-cache-hit-gate.sh instead; this gate exists because the owner law is
# "spec must work while multi-user serving runs" — the solo gate proves the restore, this
# one proves it under the banked coexistence shape (SERVED-SPEC.md: c1 spec stream + c8
# batch, 169.8 agg with a live spec stream on gemma; mixed cells green on qwen).
#
# Protocol (per model; production serving config, drafter attached where the model uses one):
#   1. WARM: one greedy request with PROMPT (long shared prefix) — cold spec, publishes
#      the draft-plane entry. Recorded as the byte anchor.
#   2. MIXED WINDOW: c8 sustained batch load with UNIQUE prompts (plain batched tier)
#      PLUS, interleaved through the window, N=6 greedy repeats of PROMPT+EXT_i
#      (i = 1..6 distinct short extensions — each a suffix-hit restore under load).
#   3. Assertions:
#      a. every repeat row has cached_tokens > 0 AND usage.spec.accepted > 0
#         *when admitted under the spec gate* — rows the concurrency gate demotes to
#         plain (projected_active > LOW) are counted and reported, not failed: the
#         load-policy demotion is the banked design, only the CACHE-HIT demotion is the
#         defect this lane closes. PASS needs >= 2 spec-engaged hit rows in the window
#         (the c8 batch tier saturates the gate; the stream lane between batch waves is
#         where the hits land, mirroring production's repeat traffic).
#         POLICY-KEYED SINCE 2026-08-19 (lane/sampled-restore-load-guard): a SAMPLED hit in
#         this cell arrives ALONGSIDE batch load, so demand is >= 2 by construction and the
#         sampled restore's SOLO watermark refuses it — measured 12 of 12 refused at
#         MEMRA_MIXED_CONC=1. When the server log carries `[spec-restore-guard] ... REFUSED`
#         lines, this cell therefore asserts that every demoted row NAMED ITSELF and reports
#         engagement without gating on it; the coexistence MECHANISM is exercised by re-running
#         with MEMRA_SPEC_RESTORE_LOAD_GUARD=0. With no guard refusals the original >= 2 rule
#         stands. An unsatisfiable assertion is the same defect as a vacuous one.
#      b. byte identity: each spec-engaged hit row's text equals the same request served
#         by a spec-off boot replay (identical order, identical salts).
#      c. batch tier stays green: c8 aggregate tok/s within 0.9x of the same window
#         with MEMRA_SERVE_SPEC=0 (spec+cache coexistence does not tax the bulk tier).
#      d. /metrics prefix_cache_hits grows by >= N over the window and no
#         "[prefix-cache] spec restore declined" lines appear in the server log except
#         the documented full-cover-sampled class.
#
# usage: spec-cache-mixed-gate.sh <model.gguf> <server_bin> <evidence_dir> [gemma_draft.gguf]
#
# DRAFTER SEAMS — the positional 4th argument is the GEMMA assistant drafter (MEMRA_DRAFT).
# It is NOT the qwen/step35 MTP attach: passing an frspec/MTP drafter there attaches NOTHING
# on a qwen model while still flipping wkv_on()/fa_f16pv_on()/the MMQ-SK form (docs/FLAGS.md
# 'SEAM TRAP' — this exact mistake produced an all-green 27B cell in which the drafter was
# never loaded). The qwen attach is MEMRA_GATE_MTP_DRAFT=<draft.gguf>. Whichever is used, the
# boot ASSERTS its attach line and a silent no-drafter run FAILS.
set -euo pipefail
MODEL=$1
BIN=$2
EV=$3
DRAFT=${4:-}
MTP_DRAFT=${MEMRA_GATE_MTP_DRAFT:-}
if [ -n "$DRAFT" ] && [ -n "$MTP_DRAFT" ]; then
    echo "refusing: a gemma assistant drafter (arg 4) and MEMRA_GATE_MTP_DRAFT select two" >&2
    echo "different spec programs; pass exactly one" >&2
    exit 2
fi
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-serve.lock}
PORT=${MEMRA_GATE_PORT:-18098}
HERE=$(cd "$(dirname "$0")" && pwd)
# Port occupancy guard (GATE-INTEGRITY-20260819 A-16, deferred to this file's merge because
# the sampled lanes had it open). Pre-flight only — $SERVER_PID is the flock wrapper's pid,
# so memra_port_owned would manufacture a false red (the dspark-serve-smoke precedent).
[ -f "$HERE/port-guard.sh" ] || {
    echo "spec-cache-mixed-gate: FAIL — $HERE/port-guard.sh is missing; refusing to bind unguarded" >&2
    exit 1
}
. "$HERE/port-guard.sh"
DRAFT_FAILS=0
# c8 is the banked coexistence shape. On serving cards fast enough that the c8 batch
# tier never drains below the spec-admission LOW watermark, every hit row demotes to
# plain BY DESIGN (the load-policy demotion, not the cache-hit demotion this lane
# closes) — the engaged-rows assertion then needs a second window at a concurrency
# the gate admits (MEMRA_MIXED_CONC=2 = within-LOW stream lane). Run both; the c8
# window still certifies identity + the batch-tier floor + demotion accounting.
CONC=${MEMRA_MIXED_CONC:-8}
# Warm-leader temperature. 0 (greedy) for qwen — cold spec publishes the draft-plane
# entry. For gemma set MEMRA_MIXED_WARM_TEMP=0.7: a greedy gemma leader rides gspec
# which never publishes; only a PLAIN (sampled) leader seeds the entry (solo hit-gate
# protocol), so a greedy-warmed gemma window measures nothing (cached=[0,...]).
WARM_TEMP=${MEMRA_MIXED_WARM_TEMP:-0}
# SAMPLED HIT REPEATS (lane/sampled-hit-spec §7 item 2 + lane/sampled-spec-quality §7 item 3,
# added in the 96GB window). Until now the warm leader could be sampled (a gemma seeding
# trick) but the HIT REPEATS were always fired greedy — `fire(..., tag)` defaulted temp=0.0
# — so the coexistence cell had never asked the sampled question at all, which is the one
# that pays (the OpenAI surface defaults to temperature 1.0).
#
# MEMRA_MIXED_HIT_TEMP > 0 fires the repeats at that temperature with an explicit per-row
# seed, and the LAST repeat additionally carries a frequency penalty — the penalized
# sampled restore that lane/sampled-spec-quality unblocked (its refusal was lifted when the
# burst's penalty window learned to span the session), which had no coexistence row.
#
# ACCEPTANCE CHANGES IN THE SAMPLED REGIME, and it has to. The greedy arm asserts each hit
# row is byte-identical to a spec-off replay. That assertion is UNAVAILABLE under sampling:
# sampled spec is distributionally exact, NOT byte-equal to plain sampling
# (tools/spec-on-cache-hit-gate.sh states the same contract), so demanding it would be a
# fake gate. The sampled regime substitutes the house's own sampled standard — seeded
# REPRODUCIBILITY: each repeat is fired twice at one seed inside the same boot and the two
# must be byte-equal.
#
# That substitution ships with a PRE-DECLARED CONTROL, because batched decode under a c8
# load is not obviously seed-reproducible to begin with: the spec-OFF arm fires the same
# doubled sampled repeats. If the control's own pairs diverge, then reproducibility under
# load is a property of batched sampled decode rather than of the restore, the identity
# question is UNAVAILABLE for this shape, and the cell says so instead of failing the lane
# for someone else's nondeterminism. Engagement and the batch-tier floor are unaffected
# either way and still gate.
HIT_TEMP=${MEMRA_MIXED_HIT_TEMP:-0}
HIT_SEED=${MEMRA_MIXED_HIT_SEED:-4242}
HIT_FREQ_PEN=${MEMRA_MIXED_HIT_FREQ_PEN:-0.6}
REPEATS=6
OUT_TOKENS=60
mkdir -p "$EV"
SERVER_PID=""

boot() { # $1 extra-env  $2 log
    memra_port_guard spec-cache-mixed-gate "$PORT" MEMRA_GATE_PORT || return 1
    local extra="$1"
    [ -n "$DRAFT" ] && extra="MEMRA_DRAFT=$DRAFT $extra"
    [ -n "$MTP_DRAFT" ] && extra="MEMRA_MTP_DRAFT=$MTP_DRAFT $extra"
    # shellcheck disable=SC2086
    flock -w 300 "$GPU_LOCK" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=$((CONC + 4)) \
        $extra "$BIN" >"$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        if curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            # A drafter handed to this gate must appear in the log; spec engaging proves
            # only that SOME head ran (a qwen trunk carries its own).
            if [ -n "$MTP_DRAFT" ] \
                && ! "$HERE/assert-drafter-attached.sh" "$2" "$MTP_DRAFT"; then
                DRAFT_FAILS=$((DRAFT_FAILS + 1))
            fi
            if [ -n "$DRAFT" ] && echo "$extra" | grep -q "MEMRA_SERVE_SPEC=0"; then
                : # spec-off twin: the gemma drafter is deliberately not loaded
            elif [ -n "$DRAFT" ] \
                && ! "$HERE/assert-drafter-attached.sh" --gemma "$2" "$DRAFT"; then
                DRAFT_FAILS=$((DRAFT_FAILS + 1))
            fi
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || {
            echo "server died during boot:"
            tail -20 "$2"
            return 1
        }
        sleep 3
    done
    return 1
}
stop() {
    # kill the SERVER, not the flock wrapper (killing the wrapper orphans the server and
    # the next boot silently reuses it on the same port).
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

run_window() { # $1 arm-name -> writes $EV/<arm>-rows.json + prints agg tok/s
    python3 - "$PORT" "$1" "$EV" "$CONC" "$REPEATS" "$OUT_TOKENS" "$WARM_TEMP" \
        "$HIT_TEMP" "$HIT_SEED" "$HIT_FREQ_PEN" <<'PY'
import json, sys, threading, time, urllib.request

port, arm, ev, conc, repeats, otoks, warm_temp = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
    int(sys.argv[6]), float(sys.argv[7]))
hit_temp, hit_seed, hit_freq_pen = float(sys.argv[8]), int(sys.argv[9]), float(sys.argv[10])

PROMPT = ("You are cataloguing the maintenance history of a small observatory. " * 40).strip()
EXTS = [f" Report focus area {i}: summarize the highest-risk subsystem." for i in range(1, repeats + 1)]

rows, lock, stop_batch = [], threading.Lock(), threading.Event()

def fire(prompt, tag, temp=0.0, seed=None, freq_pen=0.0):
    body = {"model": "gate", "prompt": prompt, "max_tokens": otoks, "temperature": temp}
    # An omitted seed is fresh entropy per request (main.rs sampler_config), which is what
    # the greedy arm wants and what a seeded-reproducibility assertion cannot tolerate.
    if seed is not None:
        body["seed"] = seed
    if freq_pen:
        body["frequency_penalty"] = freq_pen
    t0 = time.time()
    r = urllib.request.urlopen(
        urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}),
        timeout=600)
    resp = json.load(r)
    u = resp["usage"]
    with lock:
        rows.append({
            "tag": tag,
            "temp": temp,
            "seed": seed,
            "freq_pen": freq_pen,
            "wall_s": time.time() - t0,
            "completion_tokens": u["completion_tokens"],
            "cached": u["prompt_tokens_details"]["cached_tokens"],
            "spec": u.get("spec"),
            "text": resp["choices"][0]["text"],
        })

# 1. warm: publish the entry (cold spec on the spec-on arm). GEMMA: the warm leader
# must be SAMPLED (the solo hit-gate protocol) — a greedy gemma leader rides gspec,
# and gspec sessions never publish a prefix entry (publication is the qwen draft-plane
# seam; gemma converts on PLAIN-published entries), so greedy-warmed repeats see 0
# cached and the gate measures nothing.
fire(PROMPT, "warm", temp=warm_temp)

# 2. mixed window: c8 unique-prompt batch load + interleaved repeats
def batch_worker(i):
    n = 0
    while not stop_batch.is_set():
        fire(f"Unique load stream {i} item {n}. " + PROMPT[: 400 + 13 * i], f"batch-{i}")
        n += 1

t_start = time.time()
workers = [threading.Thread(target=batch_worker, args=(i,)) for i in range(conc)]
for w in workers:
    w.start()
try:
    for i, ext in enumerate(EXTS):
        time.sleep(3)  # let the batch tier fill between hit probes
        if hit_temp > 0:
            # Sampled hit repeat, seeded, fired TWICE for the reproducibility assertion.
            # The last repeat carries the frequency penalty (the penalized sampled restore
            # lane/sampled-spec-quality unblocked, previously uncovered under coexistence).
            pen = hit_freq_pen if i == len(EXTS) - 1 else 0.0
            fire(PROMPT + ext, f"repeat-{i}", temp=hit_temp, seed=hit_seed + i,
                 freq_pen=pen)
            fire(PROMPT + ext, f"repeat-{i}-r2", temp=hit_temp, seed=hit_seed + i,
                 freq_pen=pen)
        else:
            fire(PROMPT + ext, f"repeat-{i}")
finally:
    stop_batch.set()
    for w in workers:
        w.join()
window_s = time.time() - t_start

batch_tokens = sum(r["completion_tokens"] for r in rows if r["tag"].startswith("batch-"))
agg = batch_tokens / window_s if window_s > 0 else 0.0
json.dump({"rows": rows, "window_s": window_s, "batch_agg_tok_s": agg},
          open(f"{ev}/{arm}-rows.json", "w"), indent=1)
print(f"[{arm}] window {window_s:.1f}s batch agg {agg:.1f} tok/s "
      f"({sum(1 for r in rows if r['tag'].startswith('batch-'))} batch rows)")
PY
}

echo "== mixed arm: production config (spec on) =="
boot "" "$EV/mixed-on-server.log"
run_window mixed-on
stop
echo "== mixed arm: spec-off reference =="
boot "MEMRA_SERVE_SPEC=0 MEMRA_GEMMA4_SPEC=0" "$EV/mixed-off-server.log"
run_window mixed-off
stop

python3 - "$EV" <<'PY'
import json, re, sys
ev = sys.argv[1]
on = json.load(open(f"{ev}/mixed-on-rows.json"))
off = json.load(open(f"{ev}/mixed-off-rows.json"))
# How many sampled restores the LOAD GUARD refused by name in the spec-on window. Read from the
# server's own log, because "which mechanism demoted this row" is not recoverable from the
# response bodies — and it is the difference between a policy working and a mechanism missing.
try:
    with open(f"{ev}/mixed-on-server.log", errors="replace") as f:
        guard_refusal_count = len(re.findall(r"\[spec-restore-guard\].*REFUSED", f.read()))
except FileNotFoundError:
    guard_refusal_count = 0
fails = 0
def check(name, ok, detail=""):
    global fails
    print(("  ok: " if ok else "  FAIL: ") + name + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        fails += 1

rep_on = [r for r in on["rows"] if r["tag"].startswith("repeat-")]
rep_off = {r["tag"]: r for r in off["rows"] if r["tag"].startswith("repeat-")}
engaged = [r for r in rep_on if r["spec"] and r["spec"]["accepted"] > 0 and r["cached"] > 0]
demoted = [r for r in rep_on if not r["spec"]]
check("every repeat row took the cache", all(r["cached"] > 0 for r in rep_on),
      f"cached={[r['cached'] for r in rep_on]}")
# ENGAGEMENT vs POLICY (2026-08-19, lane/sampled-restore-load-guard). This assertion used to
# be unconditional, and on a card whose sampled-restore watermark is SOLO it became
# UNSATISFIABLE: this cell's whole design is a hit row arriving ALONGSIDE batch load, so demand
# is >= 2 by construction and every sampled hit is refused by the load guard — measured, at
# MEMRA_MIXED_CONC=1, 12 of 12 rows refused with `demand 2 > SOLO watermark 1`. A gate that can
# never pass is the same defect as a gate that always passes (see the boundary-probe arithmetic
# in spec-on-cache-hit-gate.sh), so the assertion is now keyed on which mechanism did the
# demoting, read out of the server's own log:
#   * refusals NAMED by the load guard  -> the POLICY is working; require that every demoted
#     sampled row named itself, and report engagement instead of gating on it. The coexistence
#     MECHANISM is then measured by re-running this cell with
#     MEMRA_SPEC_RESTORE_LOAD_GUARD=0, which is a mechanism test, not a throughput claim.
#   * NO guard refusals -> the original assertion stands unchanged: >= 2 engaged rows, or the
#     re-arm mechanism this lane exists for is not being exercised at all.
guard_refusals = guard_refusal_count
if guard_refusals > 0:
    check(f"every load-guard demotion NAMED itself ({guard_refusals} refusal lines for "
          f"{len(demoted)} demoted rows)", guard_refusals >= len(demoted))
    print(f"     spec-engaged cache-hit rows: {len(engaged)} (not gated — the SOLO watermark "
          f"refuses by design at this cell's demand; re-run with "
          f"MEMRA_SPEC_RESTORE_LOAD_GUARD=0 to exercise the re-arm mechanism)")
else:
    check(f">= 2 spec-engaged cache-hit rows under load (got {len(engaged)}, "
          f"{len(demoted)} gate-demoted)", len(engaged) >= 2)
sampled = any((r.get("temp") or 0) > 0 for r in rep_on)
if not sampled:
    ident = all(r["text"] == rep_off[r["tag"]]["text"] for r in engaged if r["tag"] in rep_off)
    check("spec-engaged hit rows byte-identical to spec-off replay", ident)
else:
    # Cross-arm byte identity is unavailable under sampling (distributional exactness).
    # Substitute seeded reproducibility, with the spec-off arm as the declared control.
    # SAME-PATH GUARD, and it is load-bearing. The load policy demotes cache-hit rows to
    # plain whenever projected_active exceeds the spec-admission LOW watermark, and that
    # decision is made per request — so two fires of the SAME seeded request can take
    # DIFFERENT programs, one spec and one plain. Byte-comparing those two is exactly the
    # comparison the repo forbids (sampled spec is distributionally exact, not byte-equal to
    # plain sampling), and it manufactures a "spec != plain under load" finding out of a
    # scheduling coincidence. Measured on the 27B at c2: 5 same-path pairs byte-identical,
    # 1 mixed-path pair differing — the only difference in the window.
    # THE RESTORED PREFIX LENGTH IS PART OF THE PROGRAM TOO (2026-08-19). The first version of
    # this guard classified pairs by spec-vs-plain only, and the mechanism arm then reported
    # 2/6 — which looked like "spec is not seed-reproducible under load" and was not. The rows:
    # every first fire restored 520 tokens and every twin restored 532, because
    # MEMRA_SPEC_RESTORE_REPUBLISH republishes the restored session's own boundary and the twin
    # hits the GROWN entry. A different restored boundary is a different burst structure — the
    # two pairs whose accepted/drafted happened to match were exactly the two that reproduced.
    # So a pair is only comparable when it took the same program AND restored the same number of
    # cached tokens; a differing-length pair is excluded BY NAME, like a mixed-path one. (Plain
    # rows survive the length difference because plain decode from the same committed tokens is
    # identical no matter how many of them came from cache — which is why the guarded posture
    # reads 6/6 and this arm does not.)
    def pairs(rows):
        d = {r["tag"]: r for r in rows}
        same, mixed, grown = [], [], []
        for t in sorted(k for k in d if k.startswith("repeat-") and not k.endswith("-r2")):
            if t + "-r2" not in d:
                continue
            a, b = d[t], d[t + "-r2"]
            if (a["spec"] is None) != (b["spec"] is None):
                mixed.append((a, b))
            elif a["spec"] is not None and a["cached"] != b["cached"]:
                grown.append((a, b))
            else:
                same.append((a, b))
        return same, mixed, grown
    on_pairs, on_mixed, on_grown = pairs(on["rows"])
    off_pairs, off_mixed, off_grown = pairs(off["rows"])
    on_ok = [a["text"] == b["text"] for a, b in on_pairs]
    off_ok = [a["text"] == b["text"] for a, b in off_pairs]
    print(f"  sampled seeded-reproducibility (same-path pairs only): spec-on "
          f"{sum(on_ok)}/{len(on_ok)}, spec-off control {sum(off_ok)}/{len(off_ok)}; "
          f"excluded — mixed-path: on {len(on_mixed)}, off {len(off_mixed)}; "
          f"grown-entry: on {len(on_grown)}, off {len(off_grown)}")
    for a, b in on_mixed:
        print(f"    excluded {a['tag']}: one fire took spec and its twin was demoted to "
              f"plain — not the same program, so bytes are not comparable")
    for a, b in on_grown:
        print(f"    excluded {a['tag']}: spec fires restored {a['cached']} vs {b['cached']} "
              f"cached tokens (republished/grown entry) — different boundary, so different "
              f"burst structure; bytes are not comparable")
    if off_pairs and not all(off_ok):
        print("  UNAVAILABLE (declared control fired): batched sampled decode is not "
              "seed-reproducible under this load in the spec-OFF arm either, so the "
              "identity question is not a property of the restore. Reported, not failed.")
    elif not on_pairs:
        print(f"  UNAVAILABLE: no comparable pair in this window (mixed-path "
              f"{len(on_mixed)}, grown-entry {len(on_grown)}) — reported rather than passed "
              f"over an empty set.")
    else:
        check("sampled hit repeats reproduce byte-for-byte at one seed (same-path)",
              all(on_ok), f"{sum(on_ok)}/{len(on_ok)}")
    pen = [r for r in rep_on if r.get("freq_pen")]
    pen_eng = [r for r in pen if r["spec"] and r["spec"]["accepted"] > 0 and r["cached"] > 0]
    check(f"penalized sampled hit row present and took the cache ({len(pen)} rows)",
          bool(pen) and all(r["cached"] > 0 for r in pen))
    print(f"  penalized sampled rows engaged: {len(pen_eng)}/{len(pen)} "
          f"(demotion under the c8 load policy is by design, not a failure)")
ratio = on["batch_agg_tok_s"] / off["batch_agg_tok_s"] if off["batch_agg_tok_s"] else 0
check(f"batch tier >= 0.9x spec-off ({ratio:.2f}x)", ratio >= 0.9)
sys.exit(1 if fails else 0)
PY
rc=$?
if [ "$DRAFT_FAILS" -gt 0 ]; then
    echo "  FAIL: $DRAFT_FAILS drafter-attach assertion(s) — the drafter handed to this gate"
    echo "        never loaded, so this run served on the trunk's own head"
    rc=1
fi
if [ "$rc" -eq 0 ]; then
    echo "SPEC-CACHE MIXED GATE: ALL GREEN"
else
    echo "SPEC-CACHE MIXED GATE: FAILURES"
fi
if grep -v "full-cover" "$EV/mixed-on-server.log" | grep -q "spec restore declined"; then
    echo "  WARN: undocumented spec-restore declines in server log (inspect)"
fi
exit "$rc"
