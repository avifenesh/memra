import json, sys, urllib.request, urllib.error
port, out_path = sys.argv[1], sys.argv[2]
CHAT = "http://127.0.0.1:%s/v1/chat/completions" % port
PROMPTS = [
    ("mul", "What is 17*23? Reply with the number only."),
    ("plan", "A train leaves at 14:05 and takes 2h40m. When does it arrive? Show the arithmetic briefly."),
    ("code", "Write a Python one-liner that returns the number of vowels in a string s."),
]
LEVELS = [None, "low", "medium", "high", "xhigh"]
REFUSE = [
    ("none", {"reasoning_effort": "none"}),
    ("minimal", {"reasoning_effort": "minimal"}),
    ("banana", {"reasoning_effort": "banana"}),
    ("enabled_false", {"reasoning": {"enabled": False}}),
    ("enable_thinking_false", {"enable_thinking": False}),
]

def post(p):
    r = urllib.request.Request(CHAT, data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(r, timeout=1200)
        return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        try:
            b = json.loads(e.read().decode() or "{}")
        except Exception:
            b = {}
        return e.code, b

rows = []

def run(tag, prompt, extra, sampled):
    p = {"model": "step37", "max_tokens": 1024,
         "messages": [{"role": "user", "content": prompt}]}
    if sampled:
        p["temperature"] = 0.5
        p["top_p"] = 0.9
    else:
        p["temperature"] = 0.0
    p.update(extra)
    code, body = post(p)
    row = {"tag": tag, "http": code, "sampled": sampled}
    if code == 200:
        ch = body["choices"][0]
        m = ch.get("message") or {}
        row.update(reasoning_chars=len(m.get("reasoning") or ""),
                   content=(m.get("content") or ""),
                   finish=ch.get("finish_reason"),
                   usage=body.get("usage"),
                   reasoning_head=(m.get("reasoning") or "")[:80])
    else:
        row["error"] = json.dumps(body)[:200]
    rows.append(row)
    return row

# refusal arms (must be named 4xx)
for name, extra in REFUSE:
    r = run("refuse/%s" % name, PROMPTS[0][1], extra, True)
    print("refuse/%s: http=%s err=%s" % (name, r["http"], (r.get("error") or "")[:110]), flush=True)

# accepted arms
for pname, prompt in PROMPTS:
    for lvl in LEVELS:
        lname = lvl or "ABSENT"
        extra = {} if lvl is None else {"reasoning_effort": lvl}
        g = run("%s/%s/greedy" % (pname, lname), prompt, extra, False)
        chars = []
        for i in range(8):
            r = run("%s/%s/s%d" % (pname, lname, i), prompt, extra, True)
            if r["http"] == 200:
                chars.append(r["reasoning_chars"])
        med = sorted(chars)[len(chars)//2] if chars else -1
        print("%s/%s: greedy_rchars=%s sampled_n=%d median_rchars=%s all=%s" %
              (pname, lname, g.get("reasoning_chars"), len(chars), med, chars), flush=True)

json.dump(rows, open(out_path, "w"), indent=1)
print("WROTE", out_path)
