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


def stream_once(msgs, sid, maxtok):
    payload = {"model": "step37", "messages": msgs, "stream": True, "max_tokens": maxtok,
               "stream_options": {"include_usage": True}}
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
    rewound = "plain-affinity: rewound to" in sl
    refused = ("plain-affinity resume failed" in sl) or ("affinity rewind failed" in sl)
    eng, walk = [], []
    suffix_tokens = None
    for l in sl.splitlines():
        if "[gemm-prime] ENGAGED" in l:
            eng.append(l.strip())
        if "[gemm-prime] WALK" in l:
            walk.append(l.strip())
        if "plain-affinity: rewound to" in l:
            p = l.split("(priming ")
            if len(p) > 1:
                try:
                    suffix_tokens = int(p[1].split()[0])
                except Exception:
                    pass

    def bases(lines):
        out = []
        for l in lines:
            for tok in l.split():
                if tok.startswith("base="):
                    try:
                        out.append(int(tok[5:]))
                    except Exception:
                        pass
        return out

    eb, wb = bases(eng), bases(walk)
    return dict(rewound=rewound, refused=refused,
                eng_fresh=sum(1 for b in eb if b == 0),
                eng_suffix=sum(1 for b in eb if b > 0),
                walk_fresh=sum(1 for b in wb if b == 0),
                walk_suffix=sum(1 for b in wb if b > 0),
                suffix_tokens=suffix_tokens, eng_lines=eng[:4], walk_lines=walk[:4])


def guarded(msgs, sid, maxtok):
    """health-gated request with a log-slice receipt."""
    if not health():
        return None, "HEALTH_FAIL"
    pos = logf.tell()
    res = stream_once(msgs, sid, maxtok)
    logf.seek(pos); sl = logf.read(); logf.seek(0, 2)
    res.update(logslice_receipts(sl))
    return res, None


def load_prompts():
    U = {1: json.load(open("/root/curve-1000.json"))["messages"][0]["content"]}
    a8 = json.load(open("/root/agentic8.json"))
    for t in range(2, 9):
        U[t] = a8[t - 2]
    return U


def msgs_through(U, A, T):
    """conversation ending at user turn T, canonical replies A[1..T-1] in between."""
    msgs = [{"role": "user", "content": U[1]}]
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
    sid = "sq-%s-t%d-s%d" % (arm, T, S)
    row = dict(arm=arm, turn=T, sample=S, sid=sid, boot_id=BOOT_ID, bin_md5=BIN_MD5,
               ts=time.time(), prefix=[])
    if arm in ("gemm", "walk"):
        for t in range(1, T):
            res, err = guarded(msgs_through(U, A, t), sid, MAXTOK_PREFIX)
            ok = (err is None) and (res["err"] is None)
            row["prefix"].append(dict(
                turn=t, ok=ok, err=err or (res and res["err"]),
                rewound=res and res["rewound"], refused=res and res["refused"],
                eng_fresh=res and res["eng_fresh"], eng_suffix=res and res["eng_suffix"],
                walk_fresh=res and res["walk_fresh"], walk_suffix=res and res["walk_suffix"],
                suffix_tokens=res and res["suffix_tokens"], ttft=res and res["ttft"]))
            if not ok:
                row.update(valid=False, invalid_reason="prefix turn %d failed: %s"
                           % (t, err or res["err"]), reasoning="", content="")
                bank_row(row)
                print("ROW %s t%d s%d INVALID (%s)" % (arm, T, S, row["invalid_reason"]),
                      flush=True)
                return
    res, err = guarded(msgs_through(U, A, T), sid, MAXTOK_EVAL)
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
               rewound=res["rewound"], refused=res["refused"],
               eng_fresh=res["eng_fresh"], eng_suffix=res["eng_suffix"],
               walk_fresh=res["walk_fresh"], walk_suffix=res["walk_suffix"],
               suffix_tokens=res["suffix_tokens"],
               eng_lines=res["eng_lines"], walk_lines=res["walk_lines"])
    # arm-validity receipts, per PLAN.md
    problems = []
    if not text.strip():
        problems.append("EMPTY")
    if res["refused"]:
        problems.append("resume refused")
    if arm == "cold":
        if res["rewound"]:
            problems.append("cold row rewound")
        if res["eng_suffix"] or res["walk_suffix"]:
            problems.append("cold row took a suffix path")
        if res["eng_fresh"] == 0:
            problems.append("cold row shows no fresh batched prime")
    elif arm == "gemm":
        if not res["rewound"]:
            problems.append("warm row did not rewind")
        if res["eng_suffix"] == 0:
            problems.append("no ENGAGED base>0 (door did not take the suffix)")
        if res["walk_suffix"]:
            problems.append("suffix fell through to the walk")
    elif arm == "walk":
        if not res["rewound"]:
            problems.append("warm row did not rewind")
        if res["walk_suffix"] == 0:
            problems.append("no WALK base>0")
        if res["eng_suffix"]:
            problems.append("suffix rode the batched entry in the walk arm")
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
