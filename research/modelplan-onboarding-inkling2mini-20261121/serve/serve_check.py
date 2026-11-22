#!/usr/bin/env python3
"""Inkling 2 Mini serve checks against a running memra-server (stdlib only).

serve_check.py BASE KEY MODEL OUT_DIR [--levels 1,2,4] [--n 8] [--max-tokens 128]

API checks (each PASS/FAIL line, and api.json):
  chat non-stream and stream (SSE frames and [DONE]), /v1/completions, greedy determinism,
  a tool call parsed into tool_calls, reasoning_effort as a number (0.5) and a name ("high"),
  refusals: a numeric string ("0.5"), an out-of-range number (1.5), an image part, a forced
  tool_choice.
Concurrency: at each level c, n greedy streamed chat requests, c at a time; every request's text
must equal the same request served alone (c=1). Metrics per level (metrics.json): TTFT, TPOT,
ITL and E2E p50/p95/p99 in ms, request and output-token throughput.
"""
import json, sys, time, threading, urllib.request, urllib.error, statistics

BASE, KEY, MODEL, OUT = sys.argv[1:5]
args = sys.argv[5:]
def opt(name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default
LEVELS = [int(x) for x in opt("--levels", "1,2,4").split(",")]
N = opt("--n", 8)
MAXTOK = opt("--max-tokens", 128)

def req(path, body, stream=False, timeout=600):
    data = json.dumps(body).encode()
    r = urllib.request.Request(BASE + path, data=data, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    try:
        resp = urllib.request.urlopen(r, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace"), []
    if not stream:
        return resp.status, resp.read().decode(), []
    events, t0 = [], time.perf_counter()
    for line in resp:
        line = line.decode().rstrip("\n")
        if line.startswith("data: "):
            events.append((time.perf_counter() - t0, line[6:]))
    return resp.status, "", events

results = []
def check(name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (": " + detail if detail else ""), flush=True)

chat = lambda msgs, **kw: dict(model=MODEL, messages=msgs, temperature=0, max_tokens=kw.pop("max_tokens", 64), **kw)
user = lambda t: [{"role": "user", "content": t}]

s, body, _ = req("/v1/chat/completions", chat(user("Name three primary colors.")))
ok = s == 200
j = json.loads(body) if ok else {}
msg = j.get("choices", [{}])[0].get("message", {}) if ok else {}
check("chat non-stream", ok and (msg.get("content") or msg.get("reasoning")), f"status {s} finish {j.get('choices',[{}])[0].get('finish_reason') if ok else body[:200]}")
first = msg
s, _, ev = req("/v1/chat/completions", chat(user("Name three primary colors."), stream=True), stream=True)
frames = [e for _, e in ev]
check("chat stream", s == 200 and frames and frames[-1] == "[DONE]", f"{len(frames)} frames")
text = ""
for f in frames[:-1]:
    d = json.loads(f)
    for c in d.get("choices", []):
        text += (c.get("delta", {}).get("content") or "")
check("stream text equals non-stream", text == (first.get("content") or ""), f"{len(text)} chars")
s, body, _ = req("/v1/chat/completions", chat(user("Name three primary colors.")))
check("greedy determinism", s == 200 and json.loads(body)["choices"][0]["message"] == first)
s, body, _ = req("/v1/completions", dict(model=MODEL, prompt="The capital of France is", max_tokens=8, temperature=0))
check("completions", s == 200 and json.loads(body)["choices"][0]["text"] != "", body[:160])
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
s, body, _ = req("/v1/chat/completions", chat(user("What is the weather in Paris right now? Use the tool."), tools=tools, max_tokens=256))
calls = json.loads(body)["choices"][0]["message"].get("tool_calls") if s == 200 else None
check("tool call parsed", bool(calls) and calls[0]["function"]["name"] == "get_weather", json.dumps(calls)[:200] if calls else body[:300])
for effort in [0.5, "high", 0.0, "none"]:
    s, body, _ = req("/v1/chat/completions", chat(user("Say hi."), reasoning_effort=effort, max_tokens=16))
    check(f"reasoning_effort {effort!r} accepted", s == 200, body[:160] if s != 200 else "")
for effort in ["0.5", 1.5, "ultra"]:
    s, body, _ = req("/v1/chat/completions", chat(user("Say hi."), reasoning_effort=effort, max_tokens=8))
    check(f"reasoning_effort {effort!r} refused", s == 400, f"status {s}: {body[:160]}")
img = [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}}]}]
s, body, _ = req("/v1/chat/completions", chat(img, max_tokens=8))
check("image part refused", 400 <= s < 500, f"status {s}: {body[:200]}")
s, body, _ = req("/v1/chat/completions", chat(user("weather?"), tools=tools, tool_choice="required", max_tokens=8))
check("forced tool_choice refused", s == 400, f"status {s}: {body[:160]}")
s, body, _ = req("/v1/chat/completions", chat(user("Name three primary colors."), numeric_class="shipped"))
check("numeric_class shipped (the default) equals the unnamed answer", s == 200 and json.loads(body)["choices"][0]["message"] == first, body[:160] if s != 200 else "")
s, body, _ = req("/v1/chat/completions", chat(user("Say hi."), numeric_class="nvfp4-rig", max_tokens=8))
check("numeric_class this boot does not serve refused", s == 400, f"status {s}: {body[:200]}")

# concurrency and metrics
prompts = [f"Write a short paragraph about topic number {i}: " + t for i, t in enumerate(
    ["rivers", "mountains", "cities", "music", "the ocean", "trees", "bridges", "libraries", "trains", "clouds"])]
prompts = (prompts * ((N // len(prompts)) + 1))[:N]
def one(p):
    body = chat(user(p), stream=True, max_tokens=MAXTOK, stream_options={"include_usage": True})
    t0 = time.perf_counter()
    s, _, ev = req("/v1/chat/completions", body, stream=True)
    times, text, reasoning, usage = [], "", "", None
    for t, f in ev:
        if f == "[DONE]":
            continue
        d = json.loads(f)
        if d.get("usage"):
            usage = d["usage"]
        for c in d.get("choices", []):
            delta = c.get("delta", {})
            piece = (delta.get("content") or "") + (delta.get("reasoning") or "")
            if piece:
                times.append(t)
                text += delta.get("content") or ""
                reasoning += delta.get("reasoning") or ""
    e2e = time.perf_counter() - t0
    return dict(status=s, text=text, reasoning=reasoning, times=times, e2e=e2e,
                tokens=(usage or {}).get("completion_tokens"))
def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
metrics, solo = {}, None
for c in LEVELS:
    out = [None] * len(prompts)
    lock = threading.Lock(); nxt = [0]
    def worker():
        while True:
            with lock:
                i = nxt[0]; nxt[0] += 1
            if i >= len(prompts):
                return
            out[i] = one(prompts[i])
    t0 = time.perf_counter()
    th = [threading.Thread(target=worker) for _ in range(c)]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.perf_counter() - t0
    ttft = [o["times"][0] * 1e3 for o in out if o["times"]]
    itl = [(b - a) * 1e3 for o in out for a, b in zip(o["times"], o["times"][1:])]
    tpot = [(o["e2e"] - o["times"][0]) * 1e3 / (o["tokens"] - 1) for o in out if o["times"] and (o["tokens"] or 0) > 1]
    e2e = [o["e2e"] * 1e3 for o in out]
    toks = sum(o["tokens"] or 0 for o in out)
    q = lambda xs: {"p50": pct(xs, .5), "p95": pct(xs, .95), "p99": pct(xs, .99)}
    metrics[c] = dict(requests=len(out), wall_s=wall, ttft_ms=q(ttft), tpot_ms=q(tpot), itl_ms=q(itl), e2e_ms=q(e2e),
                      req_per_s=len(out) / wall, out_tok_per_s=toks / wall, output_tokens=toks,
                      failures=sum(1 for o in out if o["status"] != 200))
    texts = [(o["text"], o["reasoning"]) for o in out]
    if solo is None:
        solo = texts
    same = sum(1 for a, b in zip(texts, solo) if a == b)
    check(f"c={c}: every request equals its c={LEVELS[0]} answer", same == len(out) and metrics[c]["failures"] == 0, f"{same}/{len(out)} equal")
    print(f"c={c}: {json.dumps(metrics[c])}", flush=True)
json.dump(results, open(OUT + "/api.json", "w"), indent=1)
json.dump(metrics, open(OUT + "/metrics.json", "w"), indent=1)
bad = [r for r in results if not r["ok"]]
print("serve_check: ALL PASS" if not bad else f"serve_check: {len(bad)} FAIL", flush=True)
sys.exit(1 if bad else 0)
