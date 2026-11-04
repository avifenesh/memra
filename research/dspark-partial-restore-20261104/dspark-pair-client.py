#!/usr/bin/env python3
"""Client for the DSpark strict-prefix restore cell.
usage: dspark-pair-client.py PORT turn N OUT [PRIOR_TURN_JSON...]   send turn N (prior answers rebuild history)
       dspark-pair-client.py compare EVDIR                          compare warm and cold answers, read the logs
"""
import json
import pathlib
import re
import sys
import urllib.request

SYSTEM = (
    "You are a careful reviewer of an inference engine's serving code. "
    "The operator pasted the following design notes for context.\n\n"
    + "\n".join(
        f"Note {i}: the scheduler admits a request only when its KV booking fits the device budget; "
        f"a parked session keeps its prefix entry, and a later turn that extends the same prompt "
        f"restores that entry and primes only the new suffix. Item {i} of the checklist asks whether "
        f"the restored state is byte-identical to a cold prime of the whole prompt."
        for i in range(1, 41)
    )
)
USERS = [
    "Summarize the notes in three sentences.",
    "Which note numbers mention the device budget? Answer with a short list.",
    "Now write one test idea that would catch a restore that differs from a cold prime.",
]


def ask(port, n, out, priors):
    msgs = [{"role": "system", "content": SYSTEM}]
    for i in range(n - 1):
        msgs.append({"role": "user", "content": USERS[i]})
        prior = json.loads(pathlib.Path(priors[i]).read_text())
        msgs.append({"role": "assistant", "content": prior["choices"][0]["message"].get("content") or ""})
    msgs.append({"role": "user", "content": USERS[n - 1]})
    body = {"model": "q", "messages": msgs, "max_tokens": 96, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        pathlib.Path(out).write_text(r.read().decode())


def content(p):
    m = json.loads(pathlib.Path(p).read_text())["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")


def compare(ev):
    ev = pathlib.Path(ev)
    ok = True
    for t in (2, 3):
        w, c = content(ev / f"warm-t{t}.json"), content(ev / f"cold-t{t}.json")
        eq = w == c and len(w) > 1
        ok &= eq
        print(f"turn {t}: warm==cold {eq} (len {len(w)} vs {len(c)})")
    lines = (ev / "warm-restore-lines.txt").read_text().splitlines()
    restores = [tuple(map(int, m.groups())) for l in lines
                for m in [re.search(r"DSPARK restore: (\d+) of (\d+) prompt tokens", l)] if m]
    strict = [r for r in restores if r[0] < r[1]]
    print(f"DSPARK restores {restores}; strict-prefix restores {len(strict)}")
    ok &= len(strict) >= 2
    print("DSPARK-PAIR CELL", "PASS" if ok else "FAIL")


if sys.argv[1] == "compare":
    compare(sys.argv[2])
else:
    ask(int(sys.argv[1]), int(sys.argv[3]), sys.argv[4], sys.argv[5:])
