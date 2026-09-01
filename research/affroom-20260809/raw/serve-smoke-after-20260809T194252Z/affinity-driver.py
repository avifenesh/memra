import json, sys, urllib.request
PORT, MODE, PATHF = sys.argv[1], sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{PORT}/v1/completions"
SID = "smoke-affinity"
# Control tokens are NOT required here: the explicit tier (session_id) nominates directly,
# which keeps this check independent of any particular model's chat template.
SYS = ("You are a terse assistant. Answer in one short sentence.\n\n"
       "FACTS: copies overlap with compute; pinned buffers bound host memory; "
       "bytes per token set the budget.\n\n")

def ask(prompt, n=48):
    body = {"model": "smoke", "prompt": prompt, "max_tokens": n,
            "temperature": 0, "session_id": SID}
    r = urllib.request.Request(URL, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as f:
        d = json.load(f)
    return d["choices"][0]["text"]

def render(hist):
    s = SYS
    for role, text in hist:
        s += f"{role}: {text}\n"
    return s + "assistant:"

def rewrite(text):
    """THE REWRITE CLASS: delete a span from the INTERIOR of the answer, leaving its
    boundaries intact — a think-strip in miniature. Not a tail chop (that is a plain
    prefix relation, which the pre-affinity probes already handle)."""
    if len(text) < 40:
        return text
    lo = len(text) // 3
    return text[:lo] + text[lo + len(text) // 3:]

QS = ["Why does overlapping copies with compute matter?",
      "How do pinned buffers relate to that?",
      "What sets the byte budget?",
      "Summarize all three in one sentence."]

if MODE == "record":
    hist, out = [], []
    for q in QS:
        hist.append(("user", q))
        t = ask(render(hist))
        out.append(t)
        hist.append(("assistant", rewrite(t)))
    json.dump({"hist": hist, "texts": out}, open(PATHF, "w"))
else:  # replay: same prompts, rebuilt from the RECORDED history
    rec = json.load(open(PATHF))
    hist = [tuple(x) for x in rec["hist"]]
    bad = []
    for i in range(0, len(hist), 2):
        got = ask(render(hist[:i + 1]))
        want = rec["texts"][i // 2]
        # Burst overshoot: a spec burst may stop up to K tokens past max_tokens and the two
        # arms' bursts need not align, so the shorter text being a PREFIX of the longer is
        # the same tolerance serve-st-gate check 4 applies. Anything else is a divergence.
        if not (got.startswith(want) or want.startswith(got)):
            bad.append(i // 2)
    print("MISMATCH " + ",".join(map(str, bad)) if bad else "IDENTICAL")
