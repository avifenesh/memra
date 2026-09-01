# LEG A (the task's deliverable): cold vs reused TTFT on the SAME prompt, interleaved x5.
#   Pair i sends prompt P_i twice on one session: send 1 cold, send 2 rewound. Each pair carries a
#   unique nonce and session_id so nothing leaks between pairs. cold/warm alternate throughout, so
#   box clock drift cannot favour either class.
# LEG B (diagnosis): the leg-1 growing-conversation result showed reuse SUCCEEDING with ~no TTFT
#   win. Two points fitted t = a + b*suffix with b ~ 7x the cold chunked prime's ms/token. Two
#   points fitting two unknowns is not a test, so sweep the suffix and see whether it holds.
import json, hashlib, os, re, time, urllib.request, urllib.error

port = os.environ["P"]; name = os.environ["NAME"]; logp = os.environ["LOG"]
U1 = json.load(open("/root/curve-1000.json"))["messages"][0]["content"]
URL = "http://127.0.0.1:%s/v1/chat/completions" % port
ABAKE = ("The digest is recorded and the per-stage timings are attached below for reference. ") * 4

def body(msgs, sid):
    return {"model": "step37", "messages": msgs, "stream": True,
            "max_tokens": 128, "temperature": 0, "session_id": sid}

def stream_once(payload):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; parts = []
    try:
        r = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        return (None, time.perf_counter()-t0, "", e.read().decode("utf-8","replace")[:300])
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"): continue
        s = line[5:].strip()
        if s == "[DONE]": break
        try: ch = json.loads(s)
        except Exception: continue
        d = ch.get("choices", [{}])[0].get("delta", {}) or {}
        piece = (d.get("reasoning") or "") + (d.get("content") or "")
        if piece:
            if ttft is None: ttft = time.perf_counter()-t0
            parts.append(piece)
    return (ttft, time.perf_counter()-t0, "".join(parts), None)

logf = open(logp, "r", errors="replace"); logf.seek(0, 2)
REW = re.compile(r"rewound to (\d+) of (\d+) prompt tokens \(priming (\d+) suffix")

def send(tag, msgs, sid):
    pos = logf.tell()
    ttft, total, text, err = stream_once(body(msgs, sid))
    logf.seek(pos); sl = logf.read(); logf.seek(0, 2)
    m = REW.search(sl)
    rewound = m is not None
    reused_tok, prompt_tok, suffix_tok = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)
    refused = ("plain-affinity resume failed" in sl) or ("TP KV kind mismatch" in sl) \
              or ("TP KV restore refused" in sl)
    sha = hashlib.sha256(text.encode()).hexdigest()[:16] if text.strip() else None
    r = dict(tag=tag, ttft=ttft, total=total, chars=len(text), sha=sha, rewound=rewound,
             refused=refused, reused=reused_tok, prompt=prompt_tok, suffix=suffix_tok, err=err)
    print("    %-12s ttft=%s total=%.3f chars=%d sha=%s rewound=%s reused=%d/%d suffix=%d%s"
          % (tag, ("%.3f" % ttft) if ttft is not None else "NONE", total, len(text),
             sha or "EMPTY", rewound, reused_tok, prompt_tok, suffix_tok,
             (" err=%s" % err) if err else ""))
    return r

def stats(v):
    if not v: return "n=0"
    s = sorted(v); m = s[len(s)//2] if len(s) % 2 else (s[len(s)//2-1]+s[len(s)//2])/2
    return "n=%d median=%.3f min=%.3f max=%.3f spread=%.3f" % (len(s), m, s[0], s[-1], s[-1]-s[0])

print("    === LEG A: same prompt, cold vs reused, interleaved x5 ===")
pairs = []
for i in range(1, 6):
    P = [{"role": "user", "content": "[pairA %s %02d] " % (name, i) + U1}]
    sid = "pairA-%s-%02d" % (name, i)
    c = send("A%d-cold" % i, P, sid)
    w = send("A%d-reused" % i, P, sid)
    pairs.append((i, c, w))

ok = [(i, c, w) for (i, c, w) in pairs
      if c["sha"] and w["sha"] and c["ttft"] and w["ttft"] and not c["err"] and not w["err"]]
print("    valid pairs=%d/5 (empty completions are FAILED rows; sha256('') is never a pass)" % len(ok))
wins, ratios, ident = [], [], []
for i, c, w in ok:
    print("      pair%d cold=%.3f reused=%.3f win=%+.3fs ratio=%.2fx reused_rewound=%s identity=%s"
          % (i, c["ttft"], w["ttft"], c["ttft"]-w["ttft"], c["ttft"]/w["ttft"], w["rewound"],
             "MATCH" if c["sha"] == w["sha"] else "DIFFER(%s vs %s)" % (c["sha"], w["sha"])))
    wins.append(c["ttft"]-w["ttft"]); ratios.append(c["ttft"]/w["ttft"])
    ident.append(c["sha"] == w["sha"])
print("    LEG A win seconds: %s" % stats(wins))
print("    LEG A ratio      : %s" % stats(ratios))
print("    LEG A identity   : %d/%d MATCH ; rewound %d/%d"
      % (sum(ident), len(ident), sum(1 for _, _, w in ok if w["rewound"]), len(ok)))
print("    LEG A distinct cold shas across pairs = %d (a hash that discriminates prompts)"
      % len({c["sha"] for _, c, _ in ok}))

print("    === LEG B: suffix sweep -- is reused TTFT linear in the PRIMED SUFFIX? ===")
rows = []
for k in (0, 40, 120, 300, 600):
    base = [{"role": "user", "content": "[pairB %s %04d] " % (name, k) + U1}]
    sid = "sweep-%s-%04d" % (name, k)
    send("B%04d-cold" % k, base, sid)
    if k == 0:
        msgs = base
    else:
        msgs = base + [{"role": "assistant", "content": ABAKE},
                       {"role": "user", "content": " ".join(["expand"] * k)}]
    w = send("B%04d-reused" % k, msgs, sid)
    if w["sha"] and w["ttft"] and w["rewound"]:
        rows.append(w)
print("    suffix_tokens -> reused TTFT (only rewound rows):")
for w in rows:
    print("      suffix=%4d  ttft=%.3f  ms_per_suffix_token=%.2f  reused=%d/%d"
          % (w["suffix"], w["ttft"], 1000*w["ttft"]/max(w["suffix"], 1), w["reused"], w["prompt"]))
if len(rows) >= 3:
    n = len(rows); sx = sum(w["suffix"] for w in rows); sy = sum(w["ttft"] for w in rows)
    sxx = sum(w["suffix"]**2 for w in rows); sxy = sum(w["suffix"]*w["ttft"] for w in rows)
    den = n*sxx - sx*sx
    if den:
        b = (n*sxy - sx*sy)/den; a = (sy - b*sx)/n
        ss_t = sum((w["ttft"] - sy/n)**2 for w in rows)
        ss_r = sum((w["ttft"] - (a + b*w["suffix"]))**2 for w in rows)
        print("    LEAST SQUARES on %d rewound rows: ttft = %.4fs + %.4f ms/suffix-token  (R^2=%.4f)"
              % (n, a, 1000*b, 1 - ss_r/ss_t if ss_t else float("nan")))
        print("    compare: the COLD chunked prime measured ~1.0 ms/token in the leg-1 arm.")
