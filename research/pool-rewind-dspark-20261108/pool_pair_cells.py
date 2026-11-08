#!/usr/bin/env python3
"""Warm-vs-cold pair cells for the DSpark-pool rewind (memra#1093 follow-up (a)), RTX 5090.

Copied from research/pool-spec-dspark-20261107/pool_pair_cells.py with one addition: an ENV entry
`_TEMP=x` (and `_SEED=n`) sends every turn sampled at that temperature and seed instead of greedy,
so a case can check that a rewound resume draws the cold run's sampled stream too. The rewind's
log lines are added to the patterns.

Qwen3.8-27B NVFP4 + Q5_K GGUF (MTP head), context 8,192, two sessions, prefix cache at its derived
size, switches at the device default. Route `dspark` arms the DFlash2 drafter (MEMRA_DSPARK_SPEC=1);
route `mtp` leaves it off, so the MTP spec route serves. One warm boot serves a three-turn chat in
order (each turn exactly extends the previous turn's prompt and answer); one cold boot per turn 2
and turn 3 primes that turn's prompt from scratch. Per-turn max_tokens decide the route: DFlash
prefers a cold prime over a plain prefix hit once the decode budget repays the prefill
(`dspark_cold_prime_repays_prefill`, budget >= prompt / 8), which is what sends turn 2 and turn 3
to the DSpark pool's exact-extension resume.

A chat client whose re-rendered assistant turn differs from the parked stream at the template seam
misses both pools (`dspark-reuse MISS: ... best text overlap`), so the `-raw` routes also send the
three turns as /v1/completions text, turn N+1 = turn N's prompt + its completion + a short
follow-on: the exact extension a prompt-replaying client sends, which the pools' token or text tier
takes.

usage: pool_pair_cells.py OUT CASE...   CASE = name=BINARY|ROUTE|T1,T2,T3|ENV
  ENV may carry _TEMP=x and _SEED=n (request sampling; not server environment) and _EXPECT=rewind:P:R
  (the warm boot must log at least P rewound parks, at least R resumes at the prime boundary, every
  boundary on the 32-token grid and no grid-law decline, else the case fails even when its bytes
  match) or _EXPECT=skip (it must free the session and never resume from the pool).
  ROUTE: dspark, mtp, dspark-raw or mtp-raw. ENV: space-separated VAR=VALUE for every boot of the
  case (empty = none).
Writes OUT/<name>/ (per-turn JSON, server logs), OUT/verdicts.jsonl; exit 1 if any case's bytes differ.
"""
import hashlib
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.request

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("CELL_PORT", "18175"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
DRAFT = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-DFlash2-memra"
COMMON = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
          "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
ROUTES = {"dspark": {"MEMRA_DSPARK_SPEC": "1", "MEMRA_DSPARK_DRAFT": DRAFT}, "mtp": {}}
ROUTES["dspark-raw"], ROUTES["mtp-raw"] = ROUTES["dspark"], ROUTES["mtp"]
FOLLOW = ["\n\nNote numbers that mention the device budget:", "\n\nOne test idea:"]
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
RESUME = {
    "plain_pool": r"kv-reuse: (\d+) of (\d+) prompt tokens resumed",
    "plain_rewind": r"grid-rewind: plain extension of (\d+) committed rows rewound to (\d+)",
    "spec_pool": r"spec-reuse: (\d+) committed tokens resumed",
    "spec_rewind": r"grid-rewind: spec extension of (\d+) committed rows rewound to (\d+)",
    "dspark_pool": r"dspark-reuse: (\d+) committed tokens resumed",
    "dspark_rewound_resume": r"dspark-reuse: (\d+) committed tokens resumed at the prime boundary \(parked at (\d+)\)",
    "dspark_park_rewound": r"dspark-park REWOUND \(grid law\): (\d+) rows to the prime boundary (\d+)",
    "dspark_park_skipped": r"dspark-park SKIPPED \(grid law\): a recurrent trunk's (\d+) rows",
    "dspark_decline": r"dspark-reuse DECLINED \(grid law\): the parked stream's (\d+) rows",
    "dspark_miss": r"dspark-reuse MISS: (\d+) parked, best text overlap (\d+)B of (\d+)B",
    "spec_miss": r"spec-reuse MISS[^\n]*?(\d+) parked",
    "prefix_hit": r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens",
    "dspark_restore": r"DSPARK restore: (\d+) of (\d+) prompt tokens",
    "spec_restore": r"spec restore: (\d+) of (\d+) prompt tokens",
}


def boot(log, env):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(env)
    p = subprocess.Popen([env["_BIN"]], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise SystemExit(f"server exited {p.returncode}; see {log}")
        try:
            urllib.request.urlopen(URL + "/v1/models", timeout=3).read()
            return p
        except Exception:
            time.sleep(2)
    p.kill()
    raise SystemExit("server not ready in 900 s")


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


SAMPLING = {"temperature": 0}


def ask_raw(n, answers, max_tokens, out):
    prompt = SYSTEM + "\n\nSummary of the notes:"
    for i in range(n - 1):
        prompt += answers[i] + FOLLOW[i]
    body = {"model": "q", "prompt": prompt, "max_tokens": max_tokens, **SAMPLING}
    req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        obj = json.loads(r.read().decode())
    obj["_client_elapsed_s"] = round(time.perf_counter() - t0, 4)
    out.write_text(json.dumps(obj))
    text = obj["choices"][0].get("text") or ""
    return "\x00" + text, text


def ask(n, answers, max_tokens, out):
    msgs = [{"role": "system", "content": SYSTEM}]
    for i in range(n - 1):
        msgs.append({"role": "user", "content": USERS[i]})
        msgs.append({"role": "assistant", "content": answers[i]})
    msgs.append({"role": "user", "content": USERS[n - 1]})
    body = {"model": "q", "messages": msgs, "max_tokens": max_tokens, **SAMPLING,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        obj = json.loads(r.read().decode())
    obj["_client_elapsed_s"] = round(time.perf_counter() - t0, 4)
    out.write_text(json.dumps(obj))
    m = obj["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or ""), m.get("content") or ""


def resumes(log):
    text = log.read_text(errors="replace")
    return {k: [tuple(map(int, m.groups())) for m in re.finditer(rx, text)] for k, rx in RESUME.items()}


def case(name, binary, route, maxt, extra):
    ev = OUT / name
    ev.mkdir(parents=True, exist_ok=True)
    env = dict(COMMON, **ROUTES[route], _BIN=binary)
    asker = ask_raw if route.endswith("-raw") else ask
    SAMPLING.clear()
    SAMPLING["temperature"] = 0
    expect = None
    for kv in extra.split():
        k, v = kv.split("=", 1)
        if k == "_TEMP":
            SAMPLING["temperature"] = float(v)
        elif k == "_SEED":
            SAMPLING["seed"] = int(v)
        elif k == "_EXPECT":
            expect = v
        else:
            env[k] = v
    (ev / "binary.sha").write_text(hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest()[:16] + "\n")
    p = boot(ev / "warm-server.log", env)
    warm, answers = [], []
    try:
        for t in (1, 2, 3):
            full, content = asker(t, answers, maxt[t - 1], ev / f"warm-t{t}.json")
            warm.append(full)
            answers.append(content)
    finally:
        stop(p)
    cold = {}
    for t in (2, 3):
        p = boot(ev / f"cold-t{t}-server.log", env)
        try:
            cold[t], _ = asker(t, answers, maxt[t - 1], ev / f"cold-t{t}.json")
        finally:
            stop(p)
    turns = {t: {"equal": warm[t - 1] == cold[t], "warm_len": len(warm[t - 1]) - 1, "cold_len": len(cold[t]) - 1}
             for t in (2, 3)}
    turns[1] = {"warm_sha": hashlib.sha256(warm[0].encode()).hexdigest()[:10], "warm_len": len(warm[0]) - 1}
    warm_log = resumes(ev / "warm-server.log")
    if expect and expect.startswith("rewind"):
        _, parks, resumes_min = expect.split(":")
        path_ok = (len(warm_log["dspark_park_rewound"]) >= int(parks)
                   and len(warm_log["dspark_rewound_resume"]) >= int(resumes_min)
                   and all(g % 32 == 0 for _, g in warm_log["dspark_park_rewound"])
                   and not warm_log["dspark_decline"])
    elif expect == "skip":
        path_ok = bool(warm_log["dspark_park_skipped"]) and not warm_log["dspark_pool"]
    else:
        path_ok = True
    return {"case": name, "binary": binary, "route": route, "max_tokens": maxt, "env": extra or "(device default)",
            "turns": turns, "bytes": all(turns[t]["equal"] for t in (2, 3)), "expect": expect,
            "path_ok": path_ok, "warm": warm_log}


failed = False
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    binary, route, maxt, extra = rest.split("|", 3)
    v = case(name, binary, route, [int(x) for x in maxt.split(",")], extra)
    failed |= not (v["bytes"] and v["path_ok"])
    print(json.dumps(v), flush=True)
    with open(OUT / "verdicts.jsonl", "a") as f:
        f.write(json.dumps(v) + "\n")
sys.exit(1 if failed else 0)
