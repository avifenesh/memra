#!/usr/bin/env python3
"""The six chat turns of real_turns.py rendered to ids by a running memra-server (/v1/tokenize),
in the replay's prompt format (sampled at temperature 0.9, seed 20261121, 256 tokens, end at EOS).

usage: tok_turns.py ADDR KEY MODEL REAL_TURNS_PY EOS OUT_JSON [--kwargs]
"""
import json
import runpy
import sys
import urllib.request

addr, key, model, rt, eos, out = sys.argv[1:7]
kw = "--kwargs" in sys.argv
# real_turns.py reads its argv at import: give it a harmless one and take its TURNS
saved = sys.argv
sys.argv = [rt, addr, key, model, "/dev/null", "--runs", "0"]
src = open(rt).read()
turns_src = src[src.index("TURNS = ["):src.index("\n]\n", src.index("TURNS = [")) + 3]
ns = {}
exec(turns_src, ns)
sys.argv = saved
rows = []
for name, messages, extra in ns["TURNS"]:
    body = {"model": model, "messages": messages, "add_generation_prompt": True}
    if kw and extra.get("reasoning_effort") == "no_think":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"http://{addr}/v1/tokenize", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        j = json.loads(r.read())
    ids = j.get("tokens") or j.get("ids") or j.get("token_ids")
    rows.append({"name": name, "prompt_ids": ids, "n": 256, "temperature": 0.9, "seed": 20261121,
                 "eos": int(eos), "extra": extra})
    print(name, len(ids) if ids else j)
json.dump(rows, open(out, "w"))
