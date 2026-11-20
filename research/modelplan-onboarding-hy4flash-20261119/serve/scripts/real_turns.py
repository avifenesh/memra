"""A few real chat turns through the served route: the answers, their timings and the DFly2 usage.

usage: python3 real_turns.py ADDR KEY MODEL OUT_JSON

Greedy, one request at a time. Each turn's answer and token ids are written with its wall time
and, when the route drafts, its usage's `spec` (rounds, drafted, accepted); run it against a
server with drafting on and one with it off and compare the ids (they must be equal).
"""

import json
import sys
import time
import urllib.error
import urllib.request

addr, key, model, out = sys.argv[1:5]

TURNS = [
    ("capital", [{"role": "user", "content": "What is the capital of Japan, and what is it known for? Two sentences."}],
     {"reasoning_effort": "no_think"}),
    ("code", [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number "
                                          "iteratively, with a docstring."}], {"reasoning_effort": "no_think"}),
    ("explain", [{"role": "user", "content": "Explain in a short paragraph why the sky is blue."}],
     {"reasoning_effort": "no_think"}),
    ("list", [{"role": "user", "content": "List five tips for writing clear technical documentation."}],
     {"reasoning_effort": "no_think"}),
    ("reasoning", [{"role": "user", "content": "A train leaves at 9:40 and arrives at 13:05. How long is the trip?"}],
     {"reasoning_effort": "high"}),
    ("translate", [{"role": "user", "content": "Translate to French: The meeting is moved to Thursday at noon."}],
     {"reasoning_effort": "no_think"}),
]


def post(path, body, timeout=3600):
    req = urllib.request.Request(
        f"http://{addr}{path}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), time.time() - t0


rows = []
for name, messages, extra in TURNS:
    st, raw, _ = post("/v1/tokenize", {"model": model, "messages": messages,
                                       "chat_template_kwargs": {"reasoning_effort": extra["reasoning_effort"]}})
    prompt = json.loads(raw)["tokens"] if st == 200 else None
    body = {"model": model, "prompt_ids": prompt, "max_tokens": 256, "temperature": 0}
    st, raw, dt = post("/v1/completions", body)
    j = json.loads(raw) if st == 200 else {"error": raw}
    row = {"turn": name, "status": st, "prompt_tokens": len(prompt or []), "tokens": j.get("tokens"),
           "n_tokens": j.get("n_tokens"), "stop": j.get("stop_reason"), "wall_s": round(dt, 3),
           "usage": j.get("usage"), "text": j.get("text")}
    rows.append(row)
    sp = (j.get("usage") or {}).get("spec") or j.get("spec")
    print(json.dumps({k: row[k] for k in ("turn", "status", "prompt_tokens", "n_tokens", "wall_s")} | {"spec": sp}),
          flush=True)
json.dump(rows, open(out, "w"), indent=1)
