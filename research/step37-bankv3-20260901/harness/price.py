#!/usr/bin/env python3
"""Milestone-4 pricing rep runner: REAL prompts, vendor-default sampling, capped max_tokens,
loops flagged and excluded from aggregates with the exclusion stated.

Descended from research/perf-chain-20260831/harness/digits.py, with three deliberate changes:

1. REAL PROMPTS, not the digits prompt. The perf chain used a synthetic "print 1..400" prompt
   because it wanted a deterministic 512-token completion. This lane may not: the corpus law
   is that a serving-decision cell uses real prompts (never synthetic), and the reason is
   specific to this program family — the routed-expert SWEEP is what the bank layout changes,
   and a degenerate prompt collapses the router onto a narrow expert set, so a synthetic tape
   would price a selection distribution no customer produces. One rep per agentic8 turn, so
   the same eight real prompts are priced in every arm and every boot.

2. VENDOR-DEFAULT SAMPLING, i.e. NO sampling params in the request body. models.toml's
   temperature 0.5 / top_p 0.9 (StepFun's own recommendation) governs. This is the request
   shape real traffic sends, and per the 2026-08-25 owner rule it is the only shape a serving
   claim may rest on. The greedy byte gates live in gate.py; greedy is the instrument here and
   never the priced product.

3. LOOP DETECTION. Greedy loops are the classic artifact, but a sampled tail can also
   degenerate, and a looped completion inflates BOTH tok/s and acceptance by repeating cheap
   high-accept tokens. Any rep whose tail is a repeating cycle is written to the rows file
   with looped=true and EXCLUDED from the boot median, and the exclusion is printed so it
   appears in the progress log rather than only in a post-hoc query.

Per boot: 1 smoke (spec-engagement receipt from the body, not a 200), 1 discarded warmup,
then one measured rep per prompt. Rows are JSON lines.

Usage: price.py <arm> <boot> <rows.jsonl> [corpus.json] [max_tokens]
"""
import json, os, secrets, statistics, sys, time, urllib.request

ARM, BOOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CORPUS = sys.argv[4] if len(sys.argv) > 4 else "agentic8.json"
MAXTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 512
URL = "http://127.0.0.1:18640/v1/chat/completions"
MODEL = "stepfun/step-3.7-flash"
D = "/home/ubuntu/bankv3/lane"
PROMPTS = json.load(open(os.path.join(D, "harness", CORPUS)))
receipt = dict(
    l.strip().split("=", 1)
    for l in open("%s/receipts/boot-%s.receipt" % (D, ARM))
    if "=" in l and not l.startswith(" ")
)


def looped(text):
    """True when the tail is a repeating cycle.

    Deliberately conservative and shape-based rather than threshold-tuned: scan candidate
    cycle lengths over the last 600 characters and report a loop only when one cycle tiles
    the whole window at least four times. Four repeats of a 10..80 char unit is not prose.
    Returns (bool, evidence) so the receipt can say WHY a rep was excluded — an exclusion
    without its evidence is indistinguishable from dropping an inconvenient number.
    """
    if not text:
        return False, None
    tail = text[-600:]
    for n in range(10, 81):
        if len(tail) < 4 * n:
            break
        unit = tail[-n:]
        reps = 1
        while (reps + 1) * n <= len(tail) and tail[-(reps + 1) * n:-reps * n] == unit:
            reps += 1
        if reps >= 4:
            return True, {"cycle_len": n, "repeats": reps, "unit": unit[:60]}
    return False, None


def stream_rep(rep, prompt, salt, maxtok):
    # NO sampling params: the registry's vendor defaults must be what governs, or the row is
    # not the customer shape. An explicit temperature here would silently make this a
    # different measurement from the one the serving claim needs.
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + "\n\n[session %s]" % salt}],
        "max_tokens": maxtok,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first = None
    usage = None
    fp = None
    finish = None
    nchunk = 0
    text = []
    r = urllib.request.urlopen(
        urllib.request.Request(
            URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        ),
        timeout=900,
    )
    while True:
        line = r.readline()
        if not line:
            break
        s = line.decode("utf-8", "replace").strip()
        if not s.startswith("data:"):
            continue
        p = s[5:].strip()
        if p == "[DONE]":
            continue
        try:
            j = json.loads(p)
        except Exception:
            continue
        fp = j.get("system_fingerprint") or fp
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices") or []:
            d = ch.get("delta") or {}
            # step37 is a THINKING model: its bytes arrive in reasoning_content on most turns,
            # so a content-only reader measures an empty stream and reports a false zero.
            piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
            if piece:
                nchunk += 1
                text.append(piece)
                if first is None:
                    first = time.perf_counter() - t0
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    r.close()
    wall = time.perf_counter() - t0
    u = usage or {}
    sp = u.get("spec") or {}
    ct = u.get("completion_tokens")
    pt = u.get("prompt_tokens")
    full = "".join(text)
    is_loop, evidence = looped(full)
    return {
        "arm": ARM,
        "boot": BOOT,
        "rep": rep,
        "salt": salt,
        "corpus": CORPUS,
        "prompt_index": rep - 1,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "full_tokens": ct == maxtok,
        "finish_reason": finish,
        "ttft_s": round(first, 4) if first is not None else None,
        "wall_s": round(wall, 4),
        "decode_tok_s": round((ct - 1) / (wall - first), 2)
        if (ct and first is not None and wall > first)
        else None,
        "wall_tok_s": round(ct / wall, 2) if (ct and wall > 0) else None,
        "spec_acc": sp.get("acceptance_rate"),
        "spec_rounds": sp.get("rounds"),
        "spec_drafted": sp.get("drafted"),
        "spec_accepted": sp.get("accepted"),
        "looped": is_loop,
        "loop_evidence": evidence,
        "out_chars": len(full),
        "fingerprint": fp,
        "chunks": nchunk,
        "bin_md5": receipt.get("bin_md5"),
        "boot_nonce": receipt.get("boot_nonce"),
        "built_from": receipt.get("built_from"),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# SMOKE: a spec-engagement receipt from the response body. A 200 proves a listener, not that
# the speculative path this arm is priced under actually ran (the 2026-08-25 DFlash2 lesson:
# the plain path answers fluently at half speed).
smoke = stream_rep(0, PROMPTS[0], secrets.token_hex(4), 256)
spec_expected = os.environ.get("BV3_EXPECT_SPEC", "1") == "1"
engaged = (smoke.get("spec_rounds") or 0) > 0
print(
    "SMOKE",
    "SPEC-ENGAGED" if engaged else "SPEC-MISSING",
    json.dumps({k: smoke[k] for k in ("spec_acc", "spec_rounds", "wall_tok_s", "fingerprint")}),
)
if spec_expected and not engaged:
    sys.exit(9)
if not spec_expected and engaged:
    print("GATE_FAIL: spec engaged in a spec-off arm")
    sys.exit(9)

with open(OUT, "a", buffering=1) as out:
    smoke["kind"] = "smoke"
    out.write(json.dumps(smoke) + "\n")
    w = stream_rep(0, PROMPTS[0], secrets.token_hex(4), MAXTOK)
    w["kind"] = "warmup"
    out.write(json.dumps(w) + "\n")
    print("WARMUP (discarded)", w["wall_tok_s"], "tok/s")
    kept_wall, kept_dec, dropped = [], [], []
    for rep in range(1, len(PROMPTS) + 1):
        time.sleep(1)
        row = stream_rep(rep, PROMPTS[rep - 1], secrets.token_hex(4), MAXTOK)
        row["kind"] = "rep"
        out.write(json.dumps(row) + "\n")
        mark = ""
        if row["looped"]:
            dropped.append((rep, row["loop_evidence"]))
            mark = "  LOOPED->EXCLUDED %s" % json.dumps(row["loop_evidence"])
        else:
            if row["wall_tok_s"]:
                kept_wall.append(row["wall_tok_s"])
            if row["decode_tok_s"]:
                kept_dec.append(row["decode_tok_s"])
        print(
            "rep %d wall=%s decode=%s ttft=%s ct=%s acc=%s finish=%s%s"
            % (
                rep,
                row["wall_tok_s"],
                row["decode_tok_s"],
                row["ttft_s"],
                row["completion_tokens"],
                row["spec_acc"],
                row["finish_reason"],
                mark,
            )
        )
    for rep, ev in dropped:
        print("EXCLUDED rep=%d reason=loop evidence=%s" % (rep, json.dumps(ev)))
    print(
        "BOOT_MEDIAN arm=%s boot=%s wall_tok_s=%s decode_tok_s=%s n_kept=%d n_excluded=%d"
        % (
            ARM,
            BOOT,
            round(statistics.median(kept_wall), 2) if kept_wall else None,
            round(statistics.median(kept_dec), 2) if kept_dec else None,
            len(kept_wall),
            len(dropped),
        )
    )
