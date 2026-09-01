#!/usr/bin/env python3
"""Does PREFIX-CACHE RESTORE cost this model its tool call?

Why this cell exists. On the lane-head binary, in one continuous session:
  * the FIRST (cold) greedy streaming tools request emitted a correct native tool call;
  * the very next greedy non-streaming request with the same body answered
    "It's currently 18C in Paris." with NO tool call;
  * 16 subsequent sampled reps (8 bare-default, 8 at the vendor's top_p 0.95) called the
    tool 0/16 times, and 3 of them degenerated into <tool_call><tool_call>... loops.
Greedy is deterministic, so two greedy runs of one body cannot differ unless the ENGINE STATE
differs. BOX-STATE.txt already documents that a cold prefill and a prefix-restore produce
different bytes on this model. The hypothesis: the restore path, not the sampler, is what
loses the tool call. That matters commercially, because a repeated prompt IS the sold agent
shape.

DESIGN. Two arms, identical in every way except whether the prompt can hit the prefix cache:
  COLD      : a unique nonce is embedded in each request, so every rep prefills fresh.
  RESTORED  : byte-identical prompt every rep, so reps 1..N-1 restore rep 0's prefix.
Both arms are run GREEDY (the instrument: a difference cannot be a sampling draw) and then
VENDOR-DEFAULT SAMPLED (the product). Arms are INTERLEAVED rep by rep so any drift in box
state lands on both.

usage: coldrestore.py <outdir> <reps>
"""
import json, os, sys, urllib.request, urllib.error, uuid, hashlib

OUT = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
EP = "http://127.0.0.1:18400"
MODEL = "zai/glm-5.3-flash"
os.makedirs(OUT, exist_ok=True)

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]


def ask(nonce=None):
    base = "What is the weather in Paris right now? Use the tool, then tell me in one sentence."
    return f"[session {nonce}] {base}" if nonce else base


def run(content, greedy, name):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "tools": TOOLS, "max_tokens": 400, "reasoning_effort": "low", "stream": True,
            "stream_options": {"include_usage": True}}
    if greedy:
        body["temperature"] = 0.0
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(EP + "/v1/chat/completions", data=data,
                                 headers={"content-type": "application/json"})
    acc, out, fr, usage, st, sse = {}, [], None, None, -1, []
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
                    out.append(d["content"])
                for tc in d.get("tool_calls") or []:
                    e = acc.setdefault(tc.get("index", 0),
                                       {"id": None, "name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    if tc.get("id"):
                        e["id"] = tc["id"]
                    if f.get("name"):
                        e["name"] += f["name"]
                    if f.get("arguments"):
                        e["arguments"] += f["arguments"]
    except Exception as e:
        sse.append(f"{type(e).__name__}: {e}")
    open(f"{OUT}/{name}.sse", "w").write("".join(sse))
    text = "".join(out)
    u = usage or {}
    return {"name": name, "status": st, "finish": fr,
            "tool_called": bool(acc) and fr == "tool_calls",
            "tool": acc.get(0),
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "out_sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
            "content": text[:130]}


ALL = []
for mode, greedy in [("greedy", True), ("sampled", False)]:
    print("#" * 78)
    print(f"# {mode.upper()}  ({'temperature 0 = the instrument' if greedy else 'NO sampling params = the product'})")
    print("#" * 78)
    fixed = ask()  # identical every rep -> restores
    tally = {"COLD": 0, "RESTORED": 0}
    for i in range(REPS):
        for arm in ("COLD", "RESTORED"):          # INTERLEAVED, rep by rep
            content = ask(uuid.uuid4().hex) if arm == "COLD" else fixed
            r = run(content, greedy, f"{mode}-{arm}-rep{i}")
            r["arm"] = arm; r["mode"] = mode; r["rep"] = i
            tally[arm] += r["tool_called"]
            ALL.append(r)
            print(f"  {arm:>8} rep{i}: tool_called={str(r['tool_called']):>5} "
                  f"finish={str(r['finish']):>10} cached={str(r['cached_tokens']):>4}/"
                  f"{str(r['prompt_tokens']):>4} content={r['content'][:60]!r}")
    print(f"  == {mode}: COLD {tally['COLD']}/{REPS} called the tool | "
          f"RESTORED {tally['RESTORED']}/{REPS} called the tool")
    print()

json.dump(ALL, open(f"{OUT}/coldrestore.json", "w"), indent=1)
print("COLDRESTOREDONE")
