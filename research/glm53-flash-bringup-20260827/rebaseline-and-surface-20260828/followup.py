#!/usr/bin/env python3
"""Follow-ups that the battery's failures demand. Three questions, each answered decisively.

Q1 STREAMING TOOL PARSING: the battery's streaming tools case answered "It's currently rainy
   in Paris" with no tool_call, while the identical non-streaming body called the tool. Both
   ran VENDOR-DEFAULT SAMPLED, so that is not yet evidence about the parser. Re-run the same
   body GREEDY (the instrument) to separate wiring from a sampling draw, then measure the
   tool-call RATE at the bare default vs at the vendor's published top_p.

Q2 EFFORT BYTES: /v1/responses showed cached_tokens 176/176 for BOTH no-effort and low, and 0
   for high. Cache counters are an inference; greedy output shas over an explicit effort ladder
   are a measurement.

Q3 ADMISSION: an oversize prompt returned 503 CUDA OOM instead of a named refusal. Find the
   boundary: where does it stop answering, and does it ever refuse by name?

usage: followup.py <outdir>
"""
import hashlib, json, os, sys, urllib.request, urllib.error

OUT = sys.argv[1]
EP = "http://127.0.0.1:18400"
MODEL = "zai/glm-5.3-flash"
os.makedirs(OUT, exist_ok=True)
h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]
ASK = "What is the weather in Paris right now? Use the tool, then tell me in one sentence."


def call(body, name):
    data = json.dumps(body, ensure_ascii=False).encode()
    open(f"{OUT}/{name}.request.json", "wb").write(data)
    req = urllib.request.Request(EP + "/v1/chat/completions", data=data,
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            raw = r.read().decode(); st = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode(); st = e.code
    except Exception as e:
        raw = f"{type(e).__name__}: {e}"; st = -1
    open(f"{OUT}/{name}.response.txt", "w").write(raw)
    return st, raw


def stream_call(body, name):
    """Returns (status, assembled_tool_calls, content, reasoning, finish_reason, usage)."""
    body = dict(body); body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    data = json.dumps(body, ensure_ascii=False).encode()
    open(f"{OUT}/{name}.request.json", "wb").write(data)
    req = urllib.request.Request(EP + "/v1/chat/completions", data=data,
                                 headers={"content-type": "application/json"})
    acc, content, reasoning, fr, usage, st, sse = {}, [], [], None, None, -1, []
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            st = r.status
            for rl in r:
                line = rl.decode(); sse.append(line)
                s = line.strip()
                if not s.startswith("data:"):
                    continue
                pay = s[5:].strip()
                if pay == "[DONE]":
                    break
                o = json.loads(pay)
                if o.get("usage"):
                    usage = o["usage"]
                c = (o.get("choices") or [{}])[0]
                if c.get("finish_reason"):
                    fr = c["finish_reason"]
                d = c.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"])
                if d.get("reasoning"):
                    reasoning.append(d["reasoning"])
                for tc in d.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    e = acc.setdefault(i, {"id": None, "name": "", "arguments": ""})
                    if tc.get("id"):
                        e["id"] = tc["id"]
                    f = tc.get("function") or {}
                    if f.get("name"):
                        e["name"] += f["name"]
                    if f.get("arguments"):
                        e["arguments"] += f["arguments"]
    except Exception as e:
        sse.append(f"{type(e).__name__}: {e}")
    open(f"{OUT}/{name}.response.sse", "w").write("".join(sse))
    return st, acc, "".join(content), "".join(reasoning), fr, usage


print("#" * 78)
print("Q1a  STREAMING tools, GREEDY (temperature 0) — is the STREAM PARSER wired?")
print("#" * 78)
base = {"model": MODEL, "messages": [{"role": "user", "content": ASK}],
        "tools": TOOLS, "max_tokens": 500, "reasoning_effort": "low"}
g = dict(base); g["temperature"] = 0.0
st, acc, content, reasoning, fr, usage = stream_call(g, "q1a-stream-tools-greedy")
print(f"  status={st} finish={fr!r} tool_calls={json.dumps(acc)} ")
print(f"  content={content[:160]!r}")
print(f"  usage={usage}")
STREAM_PARSER_OK = bool(acc) and fr == "tool_calls"
print(f"  => STREAMING TOOL PARSER {'WIRED' if STREAM_PARSER_OK else 'NOT EMITTING tool_calls'}")

print()
print("#" * 78)
print("Q1b  NON-STREAMING tools, GREEDY — the same body, the other transport")
print("#" * 78)
st, raw = call(g, "q1b-nonstream-tools-greedy")
r = json.loads(raw) if raw.startswith("{") else {}
ch = (r.get("choices") or [{}])[0]
tcs = (ch.get("message") or {}).get("tool_calls") or []
print(f"  status={st} finish={ch.get('finish_reason')!r} tool_calls={json.dumps(tcs)}")
print(f"  content={((ch.get('message') or {}).get('content') or '')[:160]!r}")

print()
print("#" * 78)
print("Q1c  TOOL-CALL RATE: bare default (top_p unset -> 1.0) vs vendor top_p 0.95")
print("     Same prompt, same tools, 8 reps each. VENDOR generation_config.json says")
print("     temperature 1.0 / top_p 0.95; the glm5 model pack declares sampling_defaults: None,")
print("     so an omitting client is served top_p 1.0 (unfiltered 154,880-way tail).")
print("#" * 78)
N = 8
for arm, extra in [("bare-default", {}), ("vendor-top_p-0.95", {"top_p": 0.95})]:
    called = 0
    rows = []
    for i in range(N):
        b = dict(base); b.update(extra)
        st, acc, content, reasoning, fr, usage = stream_call(b, f"q1c-{arm}-rep{i}")
        ok = bool(acc) and fr == "tool_calls"
        called += ok
        rows.append({"rep": i, "tool_called": ok, "finish": fr,
                     "name": (acc.get(0) or {}).get("name"),
                     "args": (acc.get(0) or {}).get("arguments"),
                     "content": content[:100]})
        print(f"    {arm} rep{i}: tool_called={ok} finish={fr!r} content={content[:70]!r}")
    print(f"  == {arm}: {called}/{N} requests called the tool")
    json.dump(rows, open(f"{OUT}/q1c-{arm}.json", "w"), indent=1)

print()
print("#" * 78)
print("Q2  EFFORT LADDER, GREEDY: do the rendered bytes actually change per level?")
print("    Greedy makes the OUTPUT a byte oracle; prompt_tokens + cached_tokens read the")
print("    RENDERED PROMPT. Both together decide it, rather than a cache counter alone.")
print("#" * 78)
ladder = {}
for lvl in [None, "low", "medium", "high", "max"]:
    b = {"model": MODEL, "messages": [{"role": "user", "content":
         "In one sentence: why is the sky blue?"}],
         "max_tokens": 220, "temperature": 0.0}
    if lvl:
        b["reasoning_effort"] = lvl
    name = f"q2-effort-{lvl or 'omitted'}"
    st, raw = call(b, name)
    r = json.loads(raw) if raw.startswith("{") else {}
    ch = (r.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    txt = (m.get("reasoning") or "") + (m.get("content") or "")
    u = r.get("usage") or {}
    ladder[lvl or "omitted"] = {
        "status": st, "prompt_tokens": u.get("prompt_tokens"),
        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "reasoning_chars": len(m.get("reasoning") or ""),
        "out_sha16": h(txt), "finish": ch.get("finish_reason"),
        "content": (m.get("content") or "")[:100],
    }
    print(f"  effort={str(lvl):>8}: status={st} prompt_tokens={u.get('prompt_tokens')} "
          f"reasoning_chars={len(m.get('reasoning') or '')} sha={h(txt)}")
json.dump(ladder, open(f"{OUT}/q2-effort-ladder.json", "w"), indent=1)
shas = {k: v["out_sha16"] for k, v in ladder.items()}
pt = {k: v["prompt_tokens"] for k, v in ladder.items()}
print(f"  shas: {shas}")
print(f"  prompt_tokens: {pt}")
distinct = len(set(shas.values()))
print(f"  => {distinct} distinct greedy outputs across 5 effort settings")
print(f"  => omitted == low ? {shas.get('omitted') == shas.get('low')}")
print(f"  => medium == low ? {shas.get('medium') == shas.get('low')}  (commit says medium clamps DOWN to Low)")
print(f"  => max == high ?   {shas.get('max') == shas.get('high')}")

print()
print("#" * 78)
print("Q3  ADMISSION BOUNDARY at MEMRA_CTX=8192 (caps advertise context_length 1048576)")
print("#" * 78)
adm = {}
for words in [1200, 2400, 4800, 9600, 19200]:
    big = "The quick brown fox jumps over the lazy dog. " * words
    st, raw = call({"model": MODEL, "messages": [{"role": "user", "content": big}],
                    "max_tokens": 16, "temperature": 0.0,
                    "reasoning_effort": "low"}, f"q3-ctx-{words}")
    try:
        j = json.loads(raw)
    except Exception:
        j = {}
    err = ((j.get("error") or {}).get("message") if isinstance(j, dict) else None) or ""
    u = (j.get("usage") or {}) if isinstance(j, dict) else {}
    adm[words] = {"status": st, "prompt_tokens": u.get("prompt_tokens"),
                  "error": err[:200] or None}
    print(f"  x{words:>6} ({len(big):>7} chars): status={st} prompt_tokens={u.get('prompt_tokens')} "
          f"err={err[:120]!r}")
json.dump(adm, open(f"{OUT}/q3-admission.json", "w"), indent=1)
print()
print("FOLLOWUPDONE")
