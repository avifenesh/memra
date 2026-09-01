# Deep-context degeneration attribution driver (Step-3.7-Flash, defect 7).
# Instrument lineage: research/step37-sampled-quality-20260828/sq-drive.py (stream_once,
# log-slice engagement receipts, nonce practice). All arms COLD: fresh session_id, one
# request, full history in messages. VENDOR-DEFAULT SAMPLED: no temperature/top_p in any
# payload; reasoning_effort appears ONLY on the *low arms (it is the template's own
# vendor-native control, not a sampling parameter).
#
# Modes (sys.argv[1]):
#   clean_transcript  build the content-only transcript (banked, resume-safe)
#   rows              run every arm row, interleaved round-robin, resume-safe
import json, hashlib, os, sys, time, urllib.request, urllib.error

PORT = os.environ["P"]; LOG = os.environ["LOG"]
URL = "http://127.0.0.1:%s/v1/chat/completions" % PORT
LANE = os.environ.get("LANE", os.path.dirname(os.path.abspath(__file__)))
RAW = LANE + "/raw"; GEN = LANE + "/gen"
BIN_MD5 = os.environ.get("BIN_MD5", "?")
BOOT_ID = os.environ.get("BOOT_ID", "d7boot")
MAXTOK_EVAL = 1024

os.makedirs(GEN, exist_ok=True)


def health():
    try:
        r = urllib.request.urlopen("http://127.0.0.1:%s/health" % PORT, timeout=10)
        return r.status == 200
    except Exception:
        return False


logf = open(LOG, "r", errors="replace"); logf.seek(0, 2)


def stream_once(msgs, sid, maxtok, effort=None):
    payload = {"model": "step37", "messages": msgs, "stream": True, "max_tokens": maxtok,
               "stream_options": {"include_usage": True}}
    if effort:
        payload["reasoning_effort"] = effort
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
        c0 = (ch.get("choices") or [{}])[0]
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
    rewound = "spec-affinity: rewound to" in sl
    refused = ("spec-affinity: declined" in sl) or ("affinity rewind failed" in sl)
    grew = "grew parked session" in sl
    panic = ("panicked at" in sl) or ("PANIC in the GPU worker" in sl)
    eng = [l.strip() for l in sl.splitlines() if "[gemm-prime] ENGAGED" in l]
    walk = [l.strip() for l in sl.splitlines() if "[gemm-prime] WALK" in l]
    return dict(rewound=rewound, refused=refused, grew=grew, panic=panic,
                eng_lines=eng[:4], walk_lines=walk[:4])


def guarded(msgs, sid, maxtok, effort=None):
    if not health():
        return None, "HEALTH_FAIL"
    pos = logf.tell()
    res = stream_once(msgs, sid, maxtok, effort=effort)
    logf.seek(pos); sl = logf.read(); logf.seek(0, 2)
    res.update(logslice_receipts(sl))
    return res, None


def repfrac(text):
    w = text.split()
    if len(w) < 30:
        return 0.0
    grams = [" ".join(w[i:i + 6]) for i in range(len(w) - 5)]
    return round(1.0 - len(set(grams)) / len(grams), 3)


# turn-5 (forensic-PDF) vs turn-8 (agnix marketplace listing) vocabulary probe
T5_KEYS = ["forensic", "pdf", "vat-821", "tax-registration", "vector form", "read-only reviewer"]
T8_KEYS = ["marketplace", "jetbrains", "1099", "plugin.xml", "agnix", "listing"]


def key_hits(text, keys):
    low = text.lower()
    return sum(1 for k in keys if k in low)


def load_contaminated():
    tr = json.load(open(RAW + "/transcript-contaminated.json"))
    U = {int(k): v for k, v in tr["U"].items()}
    A = {int(k): v for k, v in tr["A"].items()}
    return U, A


def msgs_through(U, A, T, nonce=""):
    msgs = [{"role": "user", "content": nonce + U[1]}]
    for t in range(2, T + 1):
        msgs.append({"role": "assistant", "content": A[t - 1]})
        msgs.append({"role": "user", "content": U[t]})
    return msgs


def spec_of(res):
    return ((res.get("usage") or {}).get("spec")) if res else None


def mode_clean_transcript():
    """Content-only transcript: per turn, vendor sampled, attempts at max_tokens
    [4096, 4096, 4096, 8192]. Accept rule (amended after the first build pass, deviation
    recorded in PLAN.md): PREFER finish=stop AND content>=200; else FIRST attempt with
    content>=200 regardless of finish (the model sometimes stops inside think with zero
    content, and sometimes overflows real content at the cap - both observed on the
    first pass, receipts in clean-transcript-build.jsonl which now banks full attempt
    texts). Resume-safe: an existing accepted turn in the bank is reused."""
    U, _ = load_contaminated()
    bank_path = RAW + "/transcript-clean.json"
    log_path = RAW + "/clean-transcript-build.jsonl"
    bank = {"U": U, "A": {}, "meta": {"bin_md5": BIN_MD5, "boot_id": BOOT_ID,
            "rule": "content-only; attempts 4096x3+8192; prefer finish=stop+content>=200, fallback any content>=200; A=content[:4000]",
            "turns": []}}
    if os.path.exists(bank_path):
        bank = json.load(open(bank_path))
        bank["U"] = {int(k): v for k, v in bank["U"].items()}
        bank["A"] = {int(k): v for k, v in bank["A"].items()}
    blog = open(log_path, "a", buffering=1)
    for t in range(1, 8):
        if t in bank["A"]:
            print("CT t%d banked, skip" % t, flush=True)
            continue
        accepted = None; fallback_b = None; fallback_c = None
        for ai, mt in enumerate([4096, 4096, 4096, 8192], 1):
            sid = "d7CT-t%d-a%d" % (t, ai)
            res, err = guarded(msgs_through(bank["U"], bank["A"], t), sid, mt)
            row = dict(turn=t, attempt=ai, maxtok=mt, err=err or (res and res["err"]),
                       finish=res and res["finish"],
                       reasoning_chars=res and len(res["reasoning"]),
                       content_chars=res and len(res["content"]),
                       ttft=res and res["ttft"], total=res and res["total"],
                       spec=spec_of(res), panic=res and res["panic"],
                       reasoning=res and res["reasoning"], content=res and res["content"])
            blog.write(json.dumps(row) + "\n")
            print("CT t%d attempt %d maxtok=%d finish=%s content=%s reasoning=%s"
                  % (t, ai, mt, row["finish"], row["content_chars"], row["reasoning_chars"]),
                  flush=True)
            ok_any = (not row["err"]) and len(res["content"].strip()) >= 200
            if ok_any and res["finish"] == "stop":
                accepted = (res, ai, mt, "stop+content")
                break
            if ok_any and fallback_b is None:
                fallback_b = (res, ai, mt, "content-any")
            # Rule C (second recorded deviation): the model repeatedly samples EOS INSIDE
            # the forced-open think and delivers a complete USER-ADDRESSED answer in the
            # reasoning channel with content=0 (turn-2 receipts, both build passes). Such
            # a reply is a finished answer in the wrong channel; it is accepted as the
            # canonical turn only when no content-bearing attempt exists, and the rule is
            # banked per turn. Unlike the prior instrument's rule this never banks a
            # TRUNCATED think (finish=stop required).
            if (not row["err"]) and res["finish"] == "stop" \
                    and len(res["content"].strip()) < 200 \
                    and len(res["reasoning"].strip()) >= 200 and fallback_c is None:
                fallback_c = (res, ai, mt, "stop-reasoning-as-answer")
        if accepted is None:
            accepted = fallback_b or fallback_c
        if accepted is None:
            print("CLEAN_TRANSCRIPT_FAIL turn=%d (no attempt produced content>=200) "
                  "- aborting rather than banking a degenerate turn" % t, flush=True)
            sys.exit(10)
        res, ai, mt, rule = accepted
        text = res["content"].strip() if rule != "stop-reasoning-as-answer" \
            else res["reasoning"].strip()
        bank["meta"]["turns"].append(dict(
            turn=t, attempt=ai, maxtok=mt, finish=res["finish"], accept_rule=rule,
            content_chars=len(res["content"]), reasoning_chars=len(res["reasoning"]),
            truncated_to_4000=len(text) > 4000,
            reasoning=res["reasoning"], content=res["content"]))
        bank["A"][t] = text[:4000]
        json.dump({"U": bank["U"], "A": bank["A"], "meta": bank["meta"]},
                  open(bank_path, "w"))
    print("CLEAN_TRANSCRIPT_DONE chars=%s" %
          {t: len(a) for t, a in sorted(bank["A"].items())}, flush=True)


def bank_row(row):
    json.dump(row, open("%s/%s-s%d.json" % (GEN, row["arm"], row["sample"]), "w"))
    slim = {k: v for k, v in row.items() if k not in ("reasoning", "content")}
    with open(LANE + "/rows.jsonl", "a") as f:
        f.write(json.dumps(slim) + "\n")


def run_row(arm, s, U, A_ctrl, A_clean):
    genp = "%s/%s-s%d.json" % (GEN, arm, s)
    if os.path.exists(genp):
        print("[row] SKIP %s s%d (banked)" % (arm, s), flush=True)
        return
    nonce = "[cell d7-%s-s%d] " % (arm, s)
    maxtok = MAXTOK_EVAL
    effort = None
    if arm == "ctrl":
        msgs = msgs_through(U, A_ctrl, 8, nonce)
    elif arm == "clean":
        msgs = msgs_through(U, A_clean, 8, nonce)
    elif arm == "cleanlow":
        msgs = msgs_through(U, A_clean, 8, nonce); effort = "low"
    elif arm == "ctrllow":
        msgs = msgs_through(U, A_ctrl, 8, nonce); effort = "low"
    elif arm == "clean4k":
        msgs = msgs_through(U, A_clean, 8, nonce); maxtok = 4096
    elif arm == "empty":
        msgs = msgs_through(U, {t: "" for t in range(1, 8)}, 8, nonce)
    elif arm == "t1":
        msgs = [{"role": "user", "content": nonce + U[1]}]
    else:
        raise SystemExit("unknown arm %s" % arm)
    sid = "d7-%s-s%d" % (arm, s)
    prompt_sha = hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:16]
    res, err = guarded(msgs, sid, maxtok, effort=effort)
    row = dict(arm=arm, sample=s, sid=sid, boot_id=BOOT_ID, bin_md5=BIN_MD5,
               ts=time.time(), nonce=nonce, maxtok=maxtok, effort=effort,
               prompt_sha16=prompt_sha)
    if err or res["err"]:
        row.update(valid=False, invalid_reason=str(err or res["err"]), reasoning="", content="")
        bank_row(row)
        print("ROW %s s%d INVALID (%s)" % (arm, s, row["invalid_reason"]), flush=True)
        return
    text = res["reasoning"] + res["content"]
    judge_text = res["content"].strip() or res["reasoning"].strip()
    row.update(reasoning=res["reasoning"], content=res["content"],
               reasoning_chars=len(res["reasoning"]), content_chars=len(res["content"]),
               finish=res["finish"], ttft=res["ttft"], total=res["total"],
               spec=spec_of(res),
               sha16=hashlib.sha256(text.encode()).hexdigest()[:16] if text.strip() else None,
               repfrac=repfrac(judge_text),
               t5_keys=key_hits(judge_text, T5_KEYS), t8_keys=key_hits(judge_text, T8_KEYS),
               rewound=res["rewound"], refused=res["refused"], grew=res["grew"],
               panic=res["panic"], eng_lines=res["eng_lines"], walk_lines=res["walk_lines"])
    problems = []
    if not text.strip():
        problems.append("EMPTY")
    if res["panic"]:
        problems.append("worker panic in request slice")
    if res["rewound"] or res["grew"]:
        problems.append("cold row reused a session")
    row["valid"] = not problems
    row["invalid_reason"] = "; ".join(problems) if problems else None
    bank_row(row)
    print("ROW %s s%d valid=%s finish=%s chars=%d(+%d content) repfrac=%.3f "
          "t5=%d t8=%d ttft=%s acc=%s"
          % (arm, s, row["valid"], res["finish"], len(res["reasoning"]),
             len(res["content"]), row["repfrac"], row["t5_keys"], row["t8_keys"],
             ("%.2f" % res["ttft"]) if res["ttft"] is not None else "NONE",
             (spec_of(res) or {}).get("acceptance_rate")), flush=True)


def mode_rows():
    U, A_ctrl = load_contaminated()
    trc = json.load(open(RAW + "/transcript-clean.json"))
    A_clean = {int(k): v for k, v in trc["A"].items()}
    assert len(A_clean) == 7, "clean transcript incomplete"
    N = {"ctrl": 8, "clean": 8, "cleanlow": 8, "ctrllow": 8, "clean4k": 4,
         "empty": 6, "t1": 6}
    order = ["ctrl", "clean", "cleanlow", "ctrllow", "clean4k", "empty", "t1"]
    for s in range(1, 9):
        for arm in order:
            if s <= N[arm]:
                run_row(arm, s, U, A_ctrl, A_clean)
    print("D7_ROWS_DONE", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "clean_transcript":
        mode_clean_transcript()
    elif mode == "rows":
        mode_rows()
    else:
        print("unknown mode %s" % mode); sys.exit(2)
