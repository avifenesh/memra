#!/usr/bin/env python3
"""Warm-vs-cold pair cells across the spec K table's cached-long rung (memra#1093 follow-up), RTX 5090.

The spec K policy picks K from the request's prompt and cached lengths (`choose_spec_k_with`): a
request with at least 1,024 prompt tokens and at least 1,024 of them restored from a cache takes
the `cached-long` rung (K=2 on an untrimmed MTP head), while its cold twin takes `cold-long` (K=3).
A resume of a long conversation (the prefix cache's spec restore, or a spec-pool rewind) therefore
runs another K than the cold prime of the same prompt. Greedy verify is exact, so greedy should not
move; a sampled verify consumes Philox draws per round, so T=1 may.

Same method as research/spec-pool-grid-20261109/spec_pool_pair_cells.py (copied from it): one warm
boot serves a three-turn conversation in order, one cold boot per turn 2 and turn 3, every warm
answer byte-equal to its cold twin. Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head, no DFlash
drafter, context 8,192, two sessions, prefix cache at its derived size, device default.

Shapes: `chat-long2` / `raw-long2`, a first turn of about 1,400 tokens (45 notes and a question),
then two short follow-ups, so turn 2 restores an entry of 1,024 tokens or more.

usage: k_pair_cells.py OUT CASE...   CASE = name=BINARY|SHAPE|T1,T2,T3|ENV
  T = max_tokens per turn, `open` = omitted. ENV: VAR=VALUE for every boot, plus request keys
  _TEMP, _SEED, _FREQ, _PRES, _USER and _EXPECT:
    _EXPECT=kdiff  some warm turn ran another K than its cold twin (the base: the rung is crossed)
    _EXPECT=ksame  every warm turn ran the K of its cold twin
Writes OUT/<name>/ and OUT/verdicts.jsonl; exit 1 if any case's bytes differ or its K check fails.
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
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
COMMON = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
          "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
USERS = {
    "short": ["List three prime numbers.", "Now three more, larger ones.",
              "Which of all six is the largest?"],
    "story": ["Write a story of about 150 words about a lighthouse keeper who finds a bottle.",
              "Continue the story for another 150 words.", "Now write a 100-word epilogue."],
    "long2": [" ".join(f"Note {i}: a parked session keeps its prefix entry, and a later turn that extends the "
                       f"same prompt restores that entry and primes only the new suffix." for i in range(1, 46))
              + "\n\nList three prime numbers.", "Now three more, larger ones.", "Which of all six is the largest?"],
    "long": [" ".join(f"Note {i}: a parked session keeps its prefix entry, and a later turn that extends the "
                      f"same prompt restores that entry and primes only the new suffix." for i in range(1, 31))
             + "\n\nList three prime numbers.", "Now three more, larger ones.", "Which of all six is the largest?"],
    "mid": ["Explain in two short paragraphs why a recurrent state cache cannot be rewound for free.",
            "Give one concrete example of a test that would catch a wrong rewind.",
            "Summarize both answers in one sentence."],
}
GEN_NOTHINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
GRID_TURN1 = 32
EXACT_SYSTEM = "Answer. Please. Please. Please. Please."
RESUME = {
    # main's exact/text tier: the parked end resumed as is
    "spec_parked_end": r"spec-reuse: (\d+) committed tokens resumed",
    "spec_affinity": r"spec-affinity: rewound to (\d+) of (\d+)",
    # the fix: an exact/text hit rewound to the session's grid checkpoint, or declined
    "spec_grid_rewind": r"grid-rewind: spec extension of (\d+) committed rows rewound to (\d+)",
    "spec_grid_decline": r"grid-rewind: spec declined",
    "spec_affinity_grid_decline": r"spec-affinity: declined \(grid law",
    "spec_park_skip": r"spec-park SKIPPED \(grid law\)",
    "prefix_hit": r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens",
    "spec_restore": r"spec restore: (\d+) of (\d+) prompt tokens",
    "spec_k": r"\[spec-k\] model=\"q\" tenant=\"[^\"]*\" K=(\d+) source=(\S+) prompt=(\d+) cached=(\d+)",
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
            time.sleep(1)
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


def post(path, body, out):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        obj = json.loads(r.read().decode())
    obj["_client_elapsed_s"] = round(time.perf_counter() - t0, 4)
    obj["_request"] = body
    out.write_text(json.dumps(obj))
    return obj


class Conv:
    """Builds turn n's request from the answers of turns 1..n-1."""

    def __init__(self, shape, req):
        self.kind, topic = shape.split("-", 1)
        self.think = topic == "think"
        self.users = list(USERS["short" if topic in ("think", "grid", "exact") else topic])
        self.req = req
        self.calibrate = topic == "grid"
        self.system = EXACT_SYSTEM if topic == "exact" else None
        if topic == "exact":
            self.users = list(USERS["short"])

    def pad_first_turn(self):
        """Pad turn 1 until the rendered prompt is GRID_TURN1 tokens (once, on the warm boot)."""
        base = self.users[0]
        for k in range(40):
            u = base + " please" * k
            if self.kind == "raw":
                body = {"model": "q", "prompt": f"<|im_start|>user\n{u}<|im_end|>\n" + GEN_NOTHINK}
            else:
                body = {"model": "q", "messages": [{"role": "user", "content": u}],
                        "chat_template_kwargs": {"enable_thinking": False}, "add_generation_prompt": True}
            req = urllib.request.Request(URL + "/v1/tokenize", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                n = json.loads(r.read().decode())["count"]
            if n == GRID_TURN1:
                self.users[0] = u
                self.calibrate = False
                return
        raise SystemExit(f"no padding of {base!r} renders {GRID_TURN1} tokens")

    def ask(self, n, answers, max_tokens, out):
        if self.calibrate:
            self.pad_first_turn()
        body = dict(self.req)
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if self.kind == "raw":
            prompt = ""
            for i in range(n):
                prompt += f"<|im_start|>user\n{self.users[i]}<|im_end|>\n" + GEN_NOTHINK
                if i < n - 1:
                    prompt += answers[i][1] + "<|im_end|>\n"
            body["prompt"] = prompt
            obj = post("/v1/completions", body, out)
            text = obj["choices"][0].get("text") or ""
            return "\x00" + text, ("", text)
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        for i in range(n):
            msgs.append({"role": "user", "content": self.users[i]})
            if i < n - 1:
                a = {"role": "assistant", "content": answers[i][1]}
                if self.think and answers[i][0]:
                    a["reasoning_content"] = answers[i][0]
                msgs.append(a)
        body["messages"] = msgs
        body["chat_template_kwargs"] = {"enable_thinking": self.think}
        obj = post("/v1/chat/completions", body, out)
        m = obj["choices"][0]["message"]
        r, c = m.get("reasoning_content") or "", m.get("content") or ""
        return r + "\x00" + c, (r, c)


def scan(log):
    text = log.read_text(errors="replace")
    def conv(g):
        return int(g) if g.isdigit() else g
    return {k: [tuple(conv(g) for g in m.groups()) for m in re.finditer(rx, text)] for k, rx in RESUME.items()}


def case(name, binary, shape, maxt, extra):
    ev = OUT / name
    ev.mkdir(parents=True, exist_ok=True)
    env = dict(COMMON, _BIN=binary)
    req = {"model": "q", "temperature": 0}
    expect = None
    for kv in extra.split():
        k, v = kv.split("=", 1)
        if k == "_TEMP":
            req["temperature"] = float(v)
        elif k == "_SEED":
            req["seed"] = int(v)
        elif k == "_FREQ":
            req["frequency_penalty"] = float(v)
        elif k == "_PRES":
            req["presence_penalty"] = float(v)
        elif k == "_USER":
            req["user"] = v
        elif k == "_EXPECT":
            expect = v
        else:
            env[k] = v
    conv = Conv(shape, req)
    (ev / "binary.sha").write_text(hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest()[:16] + "\n")
    p = boot(ev / "warm-server.log", env)
    warm, answers, ttft = [], [], {}
    try:
        for t in (1, 2, 3):
            full, parts = conv.ask(t, answers, maxt[t - 1], ev / f"warm-t{t}.json")
            warm.append(full)
            answers.append(parts)
    finally:
        stop(p)
    cold = {}
    for t in (2, 3):
        p = boot(ev / f"cold-t{t}-server.log", env)
        try:
            cold[t], _ = conv.ask(t, answers, maxt[t - 1], ev / f"cold-t{t}.json")
        finally:
            stop(p)
    for t in (2, 3):
        for arm in ("warm", "cold"):
            u = json.loads((ev / f"{arm}-t{t}.json").read_text()).get("usage") or {}
            ttft[f"{arm}-t{t}"] = {"prompt": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                                   "elapsed_s": u.get("elapsed_s")}
    turns = {t: {"equal": warm[t - 1] == cold[t], "warm_len": len(warm[t - 1]) - 1, "cold_len": len(cold[t]) - 1}
             for t in (2, 3)}
    turns[1] = {"warm_sha": hashlib.sha256(warm[0].encode()).hexdigest()[:10], "warm_len": len(warm[0]) - 1}
    w = scan(ev / "warm-server.log")
    offgrid = len(w["spec_parked_end"]) + sum(1 for pos, _ in w["spec_affinity"] if pos % 32)
    offgrid += sum(1 for _, g in w["spec_grid_rewind"] if g % 32)
    # The K each turn ran: the warm boot logs one admission line per turn, in order; each cold
    # boot logs one. Compare turn by turn.
    warm_k = w["spec_k"]
    cold_k = {t: scan(ev / f"cold-t{t}-server.log")["spec_k"] for t in (2, 3)}
    k_rows = {}
    for t in (2, 3):
        wk = warm_k[t - 1] if len(warm_k) >= t else None
        ck = cold_k[t][0] if cold_k[t] else None
        k_rows[t] = {"warm": wk, "cold": ck, "same_k": bool(wk and ck and wk[0] == ck[0])}
    kdiff = any(not k_rows[t]["same_k"] for t in (2, 3))
    if expect == "kdiff":
        path_ok = kdiff
    elif expect == "ksame":
        path_ok = not kdiff
    else:
        path_ok = True
    return {"case": name, "binary": binary, "shape": shape, "max_tokens": maxt, "env": extra or "(device default)",
            "turns": turns, "bytes": all(turns[t]["equal"] for t in (2, 3)), "expect": expect,
            "offgrid_resumes": offgrid, "k": k_rows, "path_ok": path_ok, "warm": w, "usage": ttft}


failed = False
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    binary, shape, maxt, extra = rest.split("|", 3)
    v = case(name, binary, shape, [None if x == "open" else int(x) for x in maxt.split(",")], extra)
    failed |= not (v["bytes"] and v["path_ok"])
    print(json.dumps(v), flush=True)
    with open(OUT / "verdicts.jsonl", "a") as f:
        f.write(json.dumps(v) + "\n")
sys.exit(1 if failed else 0)
