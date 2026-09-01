# Sampled-quality cell driver (gemm-prime suffix door, Step-3.7-Flash).
# Instrument lineage: research/gemm-suffix-20260828/gs-drive.py (session_id semantics,
# log-slice engagement receipts, thinking-model text = reasoning + content).
# VENDOR-DEFAULT SAMPLED: no temperature / top_p anywhere in a payload.
#
# Modes (sys.argv[1]):
#   transcript  build the canonical fixed transcript A1..A7, cold, once, banked.
#   row         one evaluated row: ARM in {cold,gemm,walk}, TURN in {4,8}, SAMPLE int.
import json, hashlib, os, sys, time, urllib.request, urllib.error

PORT = os.environ["P"]; LOG = os.environ["LOG"]
URL = "http://127.0.0.1:%s/v1/chat/completions" % PORT
SQ = "/root/sq"
BIN_MD5 = os.environ.get("BIN_MD5", "?")
BOOT_ID = os.environ.get("BOOT_ID", "?")
MAXTOK_EVAL = 1024
MAXTOK_PREFIX = 64


def health():
    try:
        r = urllib.request.urlopen("http://127.0.0.1:%s/health" % PORT, timeout=10)
        return r.status == 200
    except Exception:
        return False


logf = open(LOG, "r", errors="replace"); logf.seek(0, 2)


def stream_once(msgs, sid, maxtok, stop=None):
    payload = {"model": "step37", "messages": msgs, "stream": True, "max_tokens": maxtok,
               "stream_options": {"include_usage": True}}
    if stop:
        payload["stop"] = stop
    if sid:
        payload["session_id"] = sid
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; reason = []; content = []
    usage = None; finish = None
    try:
        r = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        return dict(ttft=None, total=time.perf_counter() - t0, reasoning="", content="",
                    usage=None, finish=None, err=e.read().decode("utf-8", "replace")[:300])
    except Exception as e:
        return dict(ttft=None, total=time.perf_counter() - t0, reasoning="", content="",
                    usage=None, finish=None, err=repr(e)[:300])
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        s = line[5:].strip()
        if s == "[DONE]":
            break
        try:
            ch = json.loads(s)
        except Exception:
            continue
        if ch.get("usage"):
            usage = ch["usage"]
        cl = ch.get("choices") or [{}]
        c0 = cl[0]
        if c0.get("finish_reason"):
            finish = c0["finish_reason"]
        d = c0.get("delta") or {}
        piece = (d.get("reasoning") or "") + (d.get("content") or "")
        if piece and ttft is None:
            ttft = time.perf_counter() - t0
        if d.get("reasoning"):
            reason.append(d["reasoning"])
        if d.get("content"):
            content.append(d["content"])
    return dict(ttft=ttft, total=time.perf_counter() - t0, reasoning="".join(reason),
                content="".join(content), usage=usage, finish=finish, err=None)


def logslice_receipts(sl):
    # Receipt strings measured on THIS branch (lane/step37-main-merge-20260828, 8695bdef4a):
    #   [worker] spec-affinity: rewound to X of Y prompt tokens (explicit; priming N suffix; ..)
    #   [worker] spec-affinity: grew parked session A -> B rows (request-owned need)
    #   [worker] spec-affinity: declined (history diverged at ..)
    #   [gemm-prime] ENGAGED t=.. base=.. seq_end=..   /  [gemm-prime] WALK t=.. base=.. seq_end=..
    # Every prime (warm or cold) may end with a SMALL trailing chunk on a base>0 line
    # (t<200, "batched prime declined" on door=0, ENGAGED on door=1), so path attribution
    # keys on BIG (t>=200) suffix lines only.
    rewound = "spec-affinity: rewound to" in sl
    refused = ("spec-affinity: declined" in sl) or ("affinity rewind failed" in sl)
    grew = "grew parked session" in sl
    panic = ("panicked at" in sl) or ("PANIC in the GPU worker" in sl)
    eng, walk = [], []
    suffix_tokens = None
    for l in sl.splitlines():
        if "[gemm-prime] ENGAGED" in l:
            eng.append(l.strip())
        if "[gemm-prime] WALK" in l:
            walk.append(l.strip())
        if "spec-affinity: rewound to" in l:
            p = l.split("; priming ")
            if len(p) > 1:
                try:
                    suffix_tokens = int(p[1].split()[0])
                except Exception:
                    pass

    def tb(lines):
        out = []
        for l in lines:
            t = base = None
            for tok in l.split():
                if tok.startswith("t="):
                    try:
                        t = int(tok[2:])
                    except Exception:
                        pass
                if tok.startswith("base="):
                    try:
                        base = int(tok[5:])
                    except Exception:
                        pass
            if t is not None and base is not None:
                out.append((t, base))
        return out

    et, wt = tb(eng), tb(walk)
    return dict(rewound=rewound, refused=refused, grew=grew, panic=panic,
                eng_fresh=sum(1 for t, b in et if b == 0),
                eng_suffix=sum(1 for t, b in et if b > 0 and t >= 200),
                eng_tail=sum(1 for t, b in et if b > 0 and t < 200),
                walk_fresh=sum(1 for t, b in wt if b == 0),
                walk_suffix=sum(1 for t, b in wt if b > 0 and t >= 200),
                walk_tail=sum(1 for t, b in wt if b > 0 and t < 200),
                suffix_tokens=suffix_tokens, eng_lines=eng[:4], walk_lines=walk[:4])


def guarded(msgs, sid, maxtok, stop=None):
    """health-gated request with a log-slice receipt."""
    if not health():
        return None, "HEALTH_FAIL"
    pos = logf.tell()
    res = stream_once(msgs, sid, maxtok, stop=stop)
    logf.seek(pos); sl = logf.read(); logf.seek(0, 2)
    res.update(logslice_receipts(sl))
    return res, None


def load_prompts():
    U = {1: json.load(open("/root/curve-1000.json"))["messages"][0]["content"]}
    a8 = json.load(open("/root/agentic8.json"))
    for t in range(2, 9):
        U[t] = a8[t - 2]
    return U


def msgs_through(U, A, T, nonce=""):
    """conversation ending at user turn T, canonical replies A[1..T-1] in between.

    nonce: short per-row tag prepended to the turn-1 user text. Needed because the
    engine's park pool nominates resume candidates by longest prefix match and holds
    only ~2 sessions; with every row sharing identical transcript bytes, a warm
    replay's pre-sized session gets out-nominated by a deeper-checkpoint leftover
    from a previous row ("declined (history diverged at X of checkpoint Y)"), the
    turn fresh-primes SMALL, and later turns grow into the panic zone (attempt-4
    cycle-1 receipt). Same practice as gs-drive.py's per-leg nonces. Every arm and
    every row carries a nonce of identical shape, so no arm is advantaged; the
    conversation is byte-identical across rows except this tag.
    """
    msgs = [{"role": "user", "content": nonce + U[1]}]
    for t in range(2, T + 1):
        msgs.append({"role": "assistant", "content": A[t - 1]})
        msgs.append({"role": "user", "content": U[t]})
    return msgs


def spec_of(res):
    return ((res.get("usage") or {}).get("spec")) if res else None


def mode_transcript():
    U = load_prompts()
    A = {}
    meta = {"bin_md5": BIN_MD5, "boot_id": BOOT_ID, "maxtok": MAXTOK_EVAL,
            "reply_rule": "content if >=200 chars else reasoning, truncated to 4000 chars",
            "turns": []}
    for t in range(1, 8):
        res, err = guarded(msgs_through(U, A, t), "sqTR-t%d" % t, MAXTOK_EVAL)
        if err or res["err"]:
            print("TRANSCRIPT_FAIL turn=%d err=%s" % (t, err or res["err"]), flush=True)
            sys.exit(10)
        if res["rewound"]:
            print("TRANSCRIPT_FAIL turn=%d rewound (not a cold prime)" % t, flush=True)
            sys.exit(11)
        # Reply rule: content when it is a real answer; a sub-200-char content fragment
        # (thinking model cut at the cap right after closing its reasoning) falls back
        # to the reasoning text so no canonical turn is degenerate.
        ctext = res["content"].strip()
        text = ctext if len(ctext) >= 200 else res["reasoning"].strip() or ctext
        if not text:
            print("TRANSCRIPT_FAIL turn=%d empty" % t, flush=True)
            sys.exit(12)
        A[t] = text[:4000]
        meta["turns"].append(dict(
            turn=t, reply_chars=len(A[t]), used_content=bool(res["content"].strip()),
            finish=res["finish"], ttft=res["ttft"], total=res["total"],
            rewound=res["rewound"], eng_fresh=res["eng_fresh"], eng_suffix=res["eng_suffix"],
            walk_fresh=res["walk_fresh"], walk_suffix=res["walk_suffix"],
            spec=spec_of(res), reasoning=res["reasoning"], content=res["content"]))
        print("TR t%d reply_chars=%d used_content=%s finish=%s ttft=%.3f eng_fresh=%d"
              % (t, len(A[t]), bool(res["content"].strip()), res["finish"],
                 res["ttft"] or -1, res["eng_fresh"]), flush=True)
    json.dump({"U": U, "A": A, "meta": meta}, open(SQ + "/transcript.json", "w"))
    print("TRANSCRIPT_DONE", flush=True)


def bank_row(row):
    genp = "%s/gen/%s-t%d-s%d.json" % (SQ, row["arm"], row["turn"], row["sample"])
    json.dump(row, open(genp, "w"))
    slim = {k: v for k, v in row.items() if k not in ("reasoning", "content", "prefix")}
    slim["n_prefix"] = len(row.get("prefix") or [])
    with open(SQ + "/rows.jsonl", "a") as f:
        f.write(json.dumps(slim) + "\n")


def mode_row():
    arm = os.environ["ARM"]; T = int(os.environ["TURN"]); S = int(os.environ["SAMPLE"])
    tr = json.load(open(SQ + "/transcript.json"))
    U = {int(k): v for k, v in tr["U"].items()}
    A = {int(k): v for k, v in tr["A"].items()}
    # SQ_WARM_SHAPE=onegrow: the sequential 8-turn replay panics the GPU worker at
    # replay turn 6 (grow bug, door-independent, see FINDINGS). Degraded-mode warm for
    # turn 8: prime the conversation through user turn T-1 in ONE fresh request, then
    # the evaluated turn is one rewind+grow+suffix-prime. Warm by construction
    # (base>0, m-fork preserved); deviation from the true turn-by-turn shape is
    # recorded on every row.
    warm_shape = os.environ.get("SQ_WARM_SHAPE", "seq")
    sid = "sq-%s-t%d-s%d" % (arm, T, S)
    row = dict(arm=arm, turn=T, sample=S, sid=sid, boot_id=BOOT_ID, bin_md5=BIN_MD5,
               warm_shape=(warm_shape if arm != "cold" else None),
               ts=time.time(), prefix=[])
    # GROW-PANIC DODGE (FINDINGS.md): spec_grow_and_rewind_to_checkpoint panics the GPU
    # worker whenever a parked session must grow past ~6-7k rows (probe: one grow to
    # ~10.2k dies; sequential grow to 6126 survives, ~7.2k dies). Session capacity is
    # fixed at creation (prompt + max_tokens + margin) and rewinds never shrink it, so
    # the FIRST prefix request carries a max_tokens that pre-sizes the session for the
    # evaluated turn's full need; every later turn rewinds + suffix-primes with NO grow.
    # Fixed-transcript seq_ends: t4 eval 4762 (+1024 gen -> need 5850), t8 eval 9118
    # (+1024 -> need 10150). Turn-1 prompt = 1480 rows.
    PRESIZE = {4: 4600, 8: 9100}
    nonce = "[cell %s-t%d-s%d] " % (arm, T, S)
    row["nonce"] = nonce
    if arm in ("gemm", "walk"):
        prefix_turns = [T - 1] if warm_shape == "onegrow" else list(range(1, T))
        for t in prefix_turns:
            # First prefix request pre-sizes capacity via max_tokens but must STOP
            # generating within ~512 tokens or the SWA ring laps its own checkpoint
            # and every later resume declines ("SWA ring lapped checkpoint", cycle-1
            # walk receipts). Capacity is reserved at admission from max_tokens, not
            # from tokens actually generated, so a stop-string keeps the reservation
            # while halting generation within a few tokens.
            first = (t == prefix_turns[0])
            mt = PRESIZE.get(T, MAXTOK_PREFIX) if first else MAXTOK_PREFIX
            res, err = guarded(msgs_through(U, A, t, nonce), sid, mt,
                               stop=(["\n", " "] if first else None))
            ok = (err is None) and (res["err"] is None)
            row["prefix"].append(dict(
                turn=t, ok=ok, err=err or (res and res["err"]),
                rewound=res and res["rewound"], refused=res and res["refused"],
                grew=res and res["grew"], panic=res and res["panic"],
                eng_fresh=res and res["eng_fresh"], eng_suffix=res and res["eng_suffix"],
                eng_tail=res and res["eng_tail"], walk_tail=res and res["walk_tail"],
                walk_fresh=res and res["walk_fresh"], walk_suffix=res and res["walk_suffix"],
                suffix_tokens=res and res["suffix_tokens"], ttft=res and res["ttft"]))
            if not ok:
                row.update(valid=False, invalid_reason="prefix turn %d failed: %s"
                           % (t, err or res["err"]), reasoning="", content="")
                bank_row(row)
                print("ROW %s t%d s%d INVALID (%s)" % (arm, T, S, row["invalid_reason"]),
                      flush=True)
                return
    res, err = guarded(msgs_through(U, A, T, nonce), sid, MAXTOK_EVAL)
    if err or res["err"]:
        row.update(valid=False, invalid_reason="eval request failed: %s" % (err or res["err"]),
                   reasoning="", content="")
        bank_row(row)
        print("ROW %s t%d s%d INVALID (%s)" % (arm, T, S, row["invalid_reason"]), flush=True)
        return
    text = (res["reasoning"] + res["content"])
    row.update(reasoning=res["reasoning"], content=res["content"],
               reasoning_chars=len(res["reasoning"]), content_chars=len(res["content"]),
               finish=res["finish"], ttft=res["ttft"], total=res["total"],
               spec=spec_of(res),
               sha16=hashlib.sha256(text.encode()).hexdigest()[:16] if text.strip() else None,
               rewound=res["rewound"], refused=res["refused"], grew=res["grew"],
               panic=res["panic"],
               eng_fresh=res["eng_fresh"], eng_suffix=res["eng_suffix"],
               eng_tail=res["eng_tail"], walk_tail=res["walk_tail"],
               walk_fresh=res["walk_fresh"], walk_suffix=res["walk_suffix"],
               suffix_tokens=res["suffix_tokens"],
               eng_lines=res["eng_lines"], walk_lines=res["walk_lines"])
    # arm-validity receipts, per PLAN.md (big-suffix lines, t>=200, attribute the path;
    # the small trailing chunk every prime emits is counted separately as *_tail)
    problems = []
    if not text.strip():
        problems.append("EMPTY")
    if res["panic"]:
        problems.append("worker panic in request slice")
    if arm == "cold":
        if res["rewound"] or res["grew"]:
            problems.append("cold row reused a session")
        if res["eng_suffix"] or res["walk_suffix"]:
            problems.append("cold row took a big suffix path")
        if res["eng_fresh"] == 0:
            problems.append("cold row shows no fresh batched prime")
    elif arm == "gemm":
        if not res["rewound"]:
            problems.append("warm row did not rewind")
        if res["eng_suffix"] == 0:
            problems.append("no ENGAGED big suffix (door did not take it)")
        if res["walk_suffix"]:
            problems.append("big suffix fell through to the walk")
    elif arm == "walk":
        if not res["rewound"]:
            problems.append("warm row did not rewind")
        if res["walk_suffix"] == 0:
            problems.append("no WALK big suffix")
        if res["eng_suffix"]:
            problems.append("big suffix rode the batched entry in the walk arm")
    if arm in ("gemm", "walk"):
        if res["grew"] or any(p.get("grew") for p in row["prefix"]):
            problems.append("a session grow happened (presize/nonce chain broken)")
    row["valid"] = not problems
    row["invalid_reason"] = "; ".join(problems) if problems else None
    bank_row(row)
    sp = spec_of(res) or {}
    print("ROW %s t%d s%d valid=%s reason=%s chars=%d(+%d content) finish=%s ttft=%s "
          "suffix=%s acc=%s eng[f=%d s=%d] walk[f=%d s=%d]"
          % (arm, T, S, row["valid"], row["invalid_reason"], len(res["reasoning"]),
             len(res["content"]), res["finish"],
             ("%.3f" % res["ttft"]) if res["ttft"] is not None else "NONE",
             res["suffix_tokens"], sp.get("acceptance"),
             res["eng_fresh"], res["eng_suffix"], res["walk_fresh"], res["walk_suffix"]),
          flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "transcript":
        mode_transcript()
    elif mode == "row":
        mode_row()
    else:
        print("unknown mode %s" % mode); sys.exit(2)
