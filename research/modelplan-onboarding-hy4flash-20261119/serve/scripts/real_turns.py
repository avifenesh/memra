"""A few real chat turns through the served route: the answers, their timings and the DFly2 usage.

usage: python3 real_turns.py ADDR KEY MODEL OUT_JSON [--greedy] [--runs N]

Chat completions, one request at a time, each stopping at its own end, sampled at the artifact's
`generation_config.json` (temperature 0.9, no top-k, no top-p) with seed 20261121, so a drafting
and a non-drafting server of one binary give the same answers, and so does every repeat.
`--runs N` asks each turn N times and keeps every wall time and their median; the answers of the
repeats must be equal. Their draft usage need not be: the draft gate prices rounds at the route's
running costs, so a repeat may draft other depths (each repeat's usage is kept in `specs`). `--greedy` asks for temperature 0: the identity and fault cells of step 10
ran that way; speed is read sampled.
"""

import json
import sys
import time
import urllib.error
import urllib.request

addr, key, model, out = sys.argv[1:5]
GREEDY = "--greedy" in sys.argv[5:]
RUNS = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 1

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
ok = True
for name, messages, extra in TURNS:
    sampling = {"temperature": 0} if GREEDY else {"temperature": 0.9, "seed": 20261121}
    body = {"model": model, "messages": messages, "max_tokens": 256} | sampling | extra
    walls, row, specs = [], None, []
    for _ in range(RUNS):
        st, raw, dt = post("/v1/chat/completions", body)
        j = json.loads(raw) if st == 200 else {"error": raw}
        ch = (j.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        usage = j.get("usage") or {}
        got = {"turn": name, "status": st, "sampling": sampling, "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens"), "finish": ch.get("finish_reason"),
               "spec": usage.get("spec"), "content": msg.get("content"),
               "reasoning": msg.get("reasoning"), "error": j.get("error")}
        answer = {k: v for k, v in got.items() if k != "spec"}
        if row is not None and answer != {k: v for k, v in row.items() if k not in ("spec", "specs")}:
            ok = False
            got["repeat_differs"] = True
        specs.append(got["spec"])
        row = got
        walls.append(round(dt, 3))
    row["walls_s"] = walls
    row["specs"] = specs
    row["wall_s"] = sorted(walls)[len(walls) // 2]
    rows.append(row)
    print(json.dumps({k: row[k] for k in ("turn", "status", "prompt_tokens", "completion_tokens", "finish",
                                          "wall_s", "walls_s", "spec")}), flush=True)
json.dump(rows, open(out, "w"), indent=1)
sys.exit(0 if ok and all(r["status"] == 200 for r in rows) else 1)
