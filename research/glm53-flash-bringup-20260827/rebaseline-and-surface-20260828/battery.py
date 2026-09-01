#!/usr/bin/env python3
# ERRATA, banked deliberately UNEDITED so this script still reproduces its banked output
# (09-battery.txt, battery-out/) byte for byte. Three of its five reported FAILs are bugs in
# THIS FILE, not server defects. See FINDINGS.txt section 4.
#   case 01  reads row["structured_output"] / row["tools"] at the TOP LEVEL; the server nests
#            both under row["capabilities"], where they read false / true, which is correct.
#            Both rows are really PASS.
#   case 05  runs the streaming tool cycle VENDOR-DEFAULT SAMPLED, so a hallucinated answer
#            instead of a tool call is a sampling draw, not a parser defect. Re-run GREEDY as
#            followup.py Q1a, where the stream assembles the tool call and PASSES.
#   case 07b asserts that reasoning effort changes input_tokens. It cannot: Low/High/Max are
#            one token each. Superseded by the greedy sha ladders (followup.py Q2 for chat,
#            respladder.py for /v1/responses), which both PASS.
# Real surface defect found by this script: case 10 only (admission not enforced before prefill).
"""Post-deploy standard-surface battery for GLM-5.3-Flash on the FIXED (lane-head) binary.

Exercises the SERVER's own rendering and parsing, not fixture bytes: every tool cycle runs a
REAL local tool whose result is fed back through the same wire format, keyed on the id and
arguments the server itself returned.

usage: battery.py <outdir>
"""
import json, os, sys, time, threading, urllib.request, urllib.error, hashlib

OUT = sys.argv[1]
EP = "http://127.0.0.1:18400"
MODEL = "zai/glm-5.3-flash"
os.makedirs(OUT, exist_ok=True)
RESULTS = []


def post(path, body, name, headers=None, timeout=900, stream=False):
    """POST and bank request+response. Returns (status, parsed_or_text, raw_text)."""
    data = json.dumps(body, ensure_ascii=False).encode()
    open(f"{OUT}/{name}.request.json", "wb").write(data)
    h = {"content-type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(EP + path, data=data, headers=h)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        status = e.code
    except Exception as e:
        raw = f"{type(e).__name__}: {e}"
        status = -1
    dt = round(time.time() - t0, 3)
    open(f"{OUT}/{name}.response.json", "w").write(raw)
    parsed = None
    if not stream:
        try:
            parsed = json.loads(raw)
        except Exception:
            pass
    return status, parsed, raw, dt


def get(path, name, timeout=60):
    try:
        with urllib.request.urlopen(EP + path, timeout=timeout) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode(); status = e.code
    except Exception as e:
        raw = f"{type(e).__name__}: {e}"; status = -1
    open(f"{OUT}/{name}.response.json", "w").write(raw)
    try:
        return status, json.loads(raw), raw
    except Exception:
        return status, None, raw


def verdict(case, ok, detail):
    RESULTS.append({"case": case, "verdict": "PASS" if ok else "FAIL", "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {case}: {detail}", flush=True)


# ---- the REAL tool ------------------------------------------------------------------
def get_weather(city):
    """The actual executed tool. Its answer is not in any prompt, so a final answer that
    contains 21 and sunny can only have come from THIS return value."""
    return {"city": city, "temp_c": 21, "sky": "sunny", "wind_kph": 11}


OAI_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "City name"}},
                       "required": ["city"]},
    },
}]
ANTH_TOOLS = [{
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {"type": "object",
                     "properties": {"city": {"type": "string"}},
                     "required": ["city"]},
}]
RESP_TOOLS = [{
    "type": "function", "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]},
}]
ASK = "What is the weather in Paris right now? Use the tool, then tell me in one sentence."

print("=" * 78)
print("CASE 1  readiness + pinned model identity  (/v1/models, /health)")
print("=" * 78)
st, models, raw = get("/v1/models", "01-models")
row = None
if models:
    for m in (models.get("data") or []):
        if m.get("id") == MODEL:
            row = m
print(json.dumps(row, indent=1)[:1200] if row else raw[:600])
verdict("01-models-identity", st == 200 and row is not None,
        f"status={st} id={MODEL} present={row is not None}")
if row is not None:
    so = row.get("structured_output")
    verdict("01-models-structured-output-honest", so is False,
            f"structured_output={so!r} (must be false: the server 400s response_format)")
    verdict("01-models-tools-advertised", bool(row.get("tools")),
            f"tools={row.get('tools')!r} context={row.get('context_length')!r}")

print()
print("=" * 78)
print("CASE 2  /v1/chat/completions  NON-STREAMING plain")
print("=" * 78)
st, r, raw, dt = post("/v1/chat/completions", {
    "model": MODEL, "messages": [{"role": "user", "content": "Say hello in exactly three words."}],
    "max_tokens": 200, "reasoning_effort": "low"}, "02-chat-plain-nonstream")
ch = (r or {}).get("choices", [{}])[0] if r else {}
msg = ch.get("message", {}) or {}
verdict("02-chat-plain-nonstream", st == 200 and bool(msg.get("content")),
        f"status={st} finish={ch.get('finish_reason')!r} "
        f"content={(msg.get('content') or '')[:80]!r} usage={(r or {}).get('usage')}")

print()
print("=" * 78)
print("CASE 3  /v1/chat/completions  STREAMING plain")
print("=" * 78)
body = {"model": MODEL, "messages": [{"role": "user", "content": "Say hello in exactly three words."}],
        "max_tokens": 200, "reasoning_effort": "low", "stream": True,
        "stream_options": {"include_usage": True}}
open(f"{OUT}/03-chat-plain-stream.request.json", "w").write(json.dumps(body))
req = urllib.request.Request(EP + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"})
chunks, content, reasoning, fr, usage, sstat = 0, [], [], None, None, -1
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        sstat = r.status
        sse = []
        for raw_line in r:
            line = raw_line.decode()
            sse.append(line)
            s = line.strip()
            if not s.startswith("data:"):
                continue
            pay = s[5:].strip()
            if pay == "[DONE]":
                break
            o = json.loads(pay)
            chunks += 1
            if o.get("usage"):
                usage = o["usage"]
            c = (o.get("choices") or [{}])[0]
            if c.get("finish_reason"):
                fr = c["finish_reason"]
            d = c.get("delta", {})
            if d.get("content"):
                content.append(d["content"])
            if d.get("reasoning"):
                reasoning.append(d["reasoning"])
        open(f"{OUT}/03-chat-plain-stream.response.sse", "w").write("".join(sse))
except Exception as e:
    open(f"{OUT}/03-chat-plain-stream.response.sse", "w").write(f"{type(e).__name__}: {e}")
verdict("03-chat-plain-stream", sstat == 200 and chunks > 1 and bool("".join(content)) and fr == "stop",
        f"status={sstat} chunks={chunks} finish={fr!r} content={''.join(content)[:80]!r} usage={usage}")

print()
print("=" * 78)
print("CASE 4  /v1/chat/completions  REAL TOOL CYCLE (non-streaming), server-rendered")
print("=" * 78)
st, r, raw, dt = post("/v1/chat/completions", {
    "model": MODEL, "messages": [{"role": "user", "content": ASK}],
    "tools": OAI_TOOLS, "max_tokens": 500, "reasoning_effort": "low"},
    "04a-chat-tools-call")
ch = (r or {}).get("choices", [{}])[0] if r else {}
msg = ch.get("message", {}) or {}
tcs = msg.get("tool_calls") or []
print(json.dumps({"finish_reason": ch.get("finish_reason"), "tool_calls": tcs}, indent=1)[:900])
ok4a = st == 200 and ch.get("finish_reason") == "tool_calls" and len(tcs) == 1 \
    and tcs[0]["function"]["name"] == "get_weather"
args = {}
if tcs:
    try:
        args = json.loads(tcs[0]["function"]["arguments"])
    except Exception:
        args = {"_unparsed": tcs[0]["function"].get("arguments")}
ok4a = ok4a and args.get("city", "").strip().lower() == "paris"
verdict("04a-chat-tools-call", ok4a,
        f"status={st} finish={ch.get('finish_reason')!r} n_calls={len(tcs)} args={args!r}")

if tcs:
    tool_out = get_weather(args.get("city", "Paris"))
    st, r2, raw, dt = post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": ASK},
            {"role": "assistant", "content": None, "tool_calls": tcs},
            {"role": "tool", "tool_call_id": tcs[0]["id"],
             "content": json.dumps(tool_out)},
        ],
        "tools": OAI_TOOLS, "max_tokens": 500, "reasoning_effort": "low"},
        "04b-chat-tools-final")
    ch2 = (r2 or {}).get("choices", [{}])[0] if r2 else {}
    final = (ch2.get("message", {}) or {}).get("content") or ""
    used = ("21" in final) and ("sunny" in final.lower())
    verdict("04b-chat-tools-final-uses-real-result", st == 200 and used and ch2.get("finish_reason") == "stop",
            f"status={st} finish={ch2.get('finish_reason')!r} used_21_and_sunny={used} "
            f"answer={final[:160]!r}")

print()
print("=" * 78)
print("CASE 5  /v1/chat/completions  REAL TOOL CYCLE (STREAMING) — delta tool_calls assembly")
print("=" * 78)
body = {"model": MODEL, "messages": [{"role": "user", "content": ASK}], "tools": OAI_TOOLS,
        "max_tokens": 500, "reasoning_effort": "low", "stream": True,
        "stream_options": {"include_usage": True}}
open(f"{OUT}/05-chat-tools-stream.request.json", "w").write(json.dumps(body))
req = urllib.request.Request(EP + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"})
acc, fr5, sstat5, sse = {}, None, -1, []
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        sstat5 = r.status
        for raw_line in r:
            line = raw_line.decode(); sse.append(line)
            s = line.strip()
            if not s.startswith("data:"):
                continue
            pay = s[5:].strip()
            if pay == "[DONE]":
                break
            o = json.loads(pay)
            c = (o.get("choices") or [{}])[0]
            if c.get("finish_reason"):
                fr5 = c["finish_reason"]
            for tc in (c.get("delta", {}) or {}).get("tool_calls", []) or []:
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
open(f"{OUT}/05-chat-tools-stream.response.sse", "w").write("".join(sse))
print(json.dumps(acc, indent=1)[:600])
sargs = {}
if acc:
    try:
        sargs = json.loads(acc[0]["arguments"])
    except Exception:
        sargs = {"_unparsed": acc.get(0, {}).get("arguments")}
verdict("05-chat-tools-stream", sstat5 == 200 and fr5 == "tool_calls" and len(acc) == 1
        and acc[0]["name"] == "get_weather" and sargs.get("city", "").lower() == "paris",
        f"status={sstat5} finish={fr5!r} assembled={acc.get(0)} args={sargs!r}")

print()
print("=" * 78)
print("CASE 6  /v1/messages (Anthropic) REAL TOOL CYCLE")
print("=" * 78)
st, r, raw, dt = post("/v1/messages", {
    "model": MODEL, "max_tokens": 500,
    "messages": [{"role": "user", "content": ASK}], "tools": ANTH_TOOLS},
    "06a-messages-tools-call", headers={"anthropic-version": "2023-06-01"})
blocks = (r or {}).get("content") or []
tu = [b for b in blocks if b.get("type") == "tool_use"]
print(json.dumps({"stop_reason": (r or {}).get("stop_reason"), "blocks": [b.get("type") for b in blocks],
                  "tool_use": tu}, indent=1)[:900])
ok6 = st == 200 and (r or {}).get("stop_reason") == "tool_use" and len(tu) == 1 \
    and tu[0].get("name") == "get_weather" \
    and str((tu[0].get("input") or {}).get("city", "")).lower() == "paris"
verdict("06a-messages-tools-call", ok6,
        f"status={st} stop_reason={(r or {}).get('stop_reason')!r} tool_use={tu[:1]}")

if tu:
    tool_out = get_weather((tu[0].get("input") or {}).get("city", "Paris"))
    st, r2, raw, dt = post("/v1/messages", {
        "model": MODEL, "max_tokens": 500,
        "messages": [
            {"role": "user", "content": ASK},
            {"role": "assistant", "content": blocks},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0]["id"],
                                          "content": json.dumps(tool_out)}]},
        ],
        "tools": ANTH_TOOLS}, "06b-messages-tools-final",
        headers={"anthropic-version": "2023-06-01"})
    txt = "".join(b.get("text", "") for b in ((r2 or {}).get("content") or [])
                  if b.get("type") == "text")
    used = ("21" in txt) and ("sunny" in txt.lower())
    verdict("06b-messages-tools-final-uses-real-result",
            st == 200 and used and (r2 or {}).get("stop_reason") in ("end_turn", "stop_sequence"),
            f"status={st} stop_reason={(r2 or {}).get('stop_reason')!r} used_21_and_sunny={used} "
            f"answer={txt[:160]!r}")

print()
print("=" * 78)
print("CASE 7  /v1/responses  REAL TOOL CYCLE + reasoning.effort RENDERS BYTES NOW")
print("=" * 78)
st, r, raw, dt = post("/v1/responses", {
    "model": MODEL, "max_output_tokens": 500, "input": ASK,
    "tools": RESP_TOOLS, "reasoning": {"effort": "low"}}, "07a-responses-effort-low")
items = (r or {}).get("output") or []
fc = [i for i in items if i.get("type") in ("function_call", "tool_call")]
print(json.dumps({"types": [i.get("type") for i in items], "fc": fc,
                  "usage": (r or {}).get("usage")}, indent=1)[:900])
verdict("07a-responses-tool-call", st == 200 and len(fc) >= 1,
        f"status={st} output_types={[i.get('type') for i in items]} n_calls={len(fc)}")

# The effort-renders-bytes proof: identical request with NO reasoning block. If effort
# reaches the template, the rendered prompt differs, so input token counts / cache hits differ.
st_noe, r_noe, raw, dt = post("/v1/responses", {
    "model": MODEL, "max_output_tokens": 16, "input": ASK, "tools": RESP_TOOLS},
    "07b-responses-no-effort")
st_low, r_low, raw, dt = post("/v1/responses", {
    "model": MODEL, "max_output_tokens": 16, "input": ASK, "tools": RESP_TOOLS,
    "reasoning": {"effort": "low"}}, "07c-responses-effort-low-short")
st_max, r_max, raw, dt = post("/v1/responses", {
    "model": MODEL, "max_output_tokens": 16, "input": ASK, "tools": RESP_TOOLS,
    "reasoning": {"effort": "high"}}, "07d-responses-effort-high-short")


def uin(x):
    u = (x or {}).get("usage") or {}
    det = u.get("input_tokens_details") or {}
    return u.get("input_tokens"), det.get("cached_tokens")


n_none, c_none = uin(r_noe)
n_low, c_low = uin(r_low)
n_high, c_high = uin(r_max)
print(f"  no-effort : input_tokens={n_none} cached={c_none}")
print(f"  effort低low: input_tokens={n_low} cached={c_low}")
print(f"  effort high: input_tokens={n_high} cached={c_high}")
differs = (n_low is not None and n_none is not None and n_low != n_none) or \
          (n_high is not None and n_low is not None and n_high != n_low)
verdict("07b-effort-renders-bytes", differs,
        f"input_tokens none={n_none} low={n_low} high={n_high}; "
        f"cached none={c_none} low={c_low} high={c_high} "
        f"(pre-fix defect was cached_tokens == full input on the effort request)")

print()
print("=" * 78)
print("CASE 8  named refusals (the off-requests this surface must NOT fake)")
print("=" * 78)
for name, body, why in [
    ("08a-response-format-refused",
     {"model": MODEL, "messages": [{"role": "user", "content": "give me json"}],
      "response_format": {"type": "json_object"}, "max_tokens": 20},
     "response_format unsupported"),
    ("08b-effort-none-refused",
     {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
      "reasoning_effort": "none", "max_tokens": 5},
     "this template has no no-think arm"),
    ("08c-effort-invalid-refused",
     {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
      "reasoning_effort": "turbo", "max_tokens": 5},
     "out-of-table effort level"),
    ("08d-unknown-model-refused",
     {"model": "zai/does-not-exist", "messages": [{"role": "user", "content": "hi"}],
      "max_tokens": 5},
     "unroutable model"),
]:
    st, r, raw, dt = post("/v1/chat/completions", body, name)
    m = ((r or {}).get("error") or {}).get("message") if isinstance(r, dict) else None
    verdict(name, st == 400 or st == 404,
            f"status={st} why={why} message={(m or raw)[:180]!r}")

print()
print("=" * 78)
print("CASE 9  concurrency: 5 in flight against MEMRA_MAX_SESSIONS=4")
print("=" * 78)
conc = {}


def worker(i):
    st, r, raw, dt = post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Count from {i} to {i+4}, digits only."}],
        "max_tokens": 120, "reasoning_effort": "low"}, f"09-conc-{i}")
    ch = (r or {}).get("choices", [{}])[0] if r else {}
    conc[i] = {"status": st, "finish": ch.get("finish_reason"),
               "content": ((ch.get("message") or {}).get("content") or "")[:60],
               "wall_s": dt}


ts = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
t0 = time.time()
for t in ts:
    t.start()
for t in ts:
    t.join()
wall = round(time.time() - t0, 2)
print(json.dumps(conc, indent=1)[:1200])
allok = all(v["status"] == 200 and v["content"] for v in conc.values())
verdict("09-concurrency-5-over-4-sessions", allok,
        f"wall={wall}s statuses={[v['status'] for v in conc.values()]} "
        f"(a 5th request must QUEUE, never fail or be dropped)")

print()
print("=" * 78)
print("CASE 10  context / admission limit at MEMRA_CTX=8192")
print("=" * 78)
big = "The quick brown fox jumps over the lazy dog. " * 3000  # ~ 30k+ tokens
st, r, raw, dt = post("/v1/chat/completions", {
    "model": MODEL, "messages": [{"role": "user", "content": big}],
    "max_tokens": 32, "reasoning_effort": "low"}, "10-oversize-context")
m = ((r or {}).get("error") or {}).get("message") if isinstance(r, dict) else None
verdict("10-oversize-context-named-refusal", st == 400,
        f"status={st} chars={len(big)} message={(m or raw)[:220]!r} "
        f"(must be a NAMED refusal, not a truncation or a hang)")

print()
print("=" * 78)
print("CASE 11  VENDOR-DEFAULT SAMPLED: is a bare request actually sampled?")
print("=" * 78)
bare = {"model": MODEL, "messages": [{"role": "user", "content":
        "Name three colours and one reason each, briefly."}],
        "max_tokens": 300, "reasoning_effort": "low"}


def txt_of(r):
    return ((r or {}).get("choices", [{}])[0].get("message") or {}).get("content") or ""


st1, r1, _, _ = post("/v1/chat/completions", dict(bare), "11a-bare-run1")
st2, r2, _, _ = post("/v1/chat/completions", dict(bare), "11b-bare-run2")
greedy = dict(bare); greedy["temperature"] = 0.0
st3, r3, _, _ = post("/v1/chat/completions", greedy, "11c-explicit-greedy")
st4, r4, _, _ = post("/v1/chat/completions", greedy, "11d-explicit-greedy-again")
h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]
t1, t2, t3, t4 = txt_of(r1), txt_of(r2), txt_of(r3), txt_of(r4)
print(f"  bare run1 sha={h(t1)}  {t1[:70]!r}")
print(f"  bare run2 sha={h(t2)}  {t2[:70]!r}")
print(f"  temp0 run1 sha={h(t3)}")
print(f"  temp0 run2 sha={h(t4)}")
verdict("11-bare-request-is-sampled-not-greedy", t1 and t2 and h(t1) != h(t2),
        f"two identical bare requests differ: {h(t1)} vs {h(t2)} "
        f"(identical would mean the omitting client is served GREEDY)")
verdict("11-explicit-greedy-is-deterministic", t3 and h(t3) == h(t4),
        f"temperature:0 reproduces: {h(t3)} == {h(t4)} (the instrument still works)")

print()
print("=" * 78)
json.dump(RESULTS, open(f"{OUT}/VERDICTS.json", "w"), indent=1)
npass = sum(1 for r in RESULTS if r["verdict"] == "PASS")
print(f"BATTERY: {npass}/{len(RESULTS)} PASS")
for r in RESULTS:
    if r["verdict"] == "FAIL":
        print(f"  FAIL {r['case']}: {r['detail']}")
