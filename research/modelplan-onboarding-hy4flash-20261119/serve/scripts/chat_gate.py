#!/usr/bin/env python3
"""The hy_v4_flash route's served-answer gates against a running memra-server (memra's own
stream shape: `MEMRA_COMPAT` unset, so /v1/completions answers the generated token ids).

usage: chat_gate.py ADDR API_KEY MODEL REQUESTS_DIR TAPES_JSON ROUTE_GEN_JSON OUT_JSON SERVER_LOG [--quick]

REQUESTS_DIR  the checkpoint requests (`chat.json`, `long.json`, `long-split.json`: token ids and
              the prefill end)
TAPES_JSON    per request the verified tape's argmax row (`{"chat": [...], "long": [...],
              "long-split": [...]}`, from the step 8 tapes the checkpoint gate passed)
ROUTE_GEN_JSON per request the GPU route's own greedy ids after the prefill (`hy4f-gpu --gen`)

Gates:
  render      /v1/tokenize of the probes' messages (reasoning off) equals the requests' prompt ids
  route       greedy /v1/completions on each request's prompt ids: the answer's ids equal the
              route's greedy ids up to the end-of-sentence id, which ends the answer (finish stop)
  forced      every teacher-forced position of the tapes' decode as its own prompt (max_tokens 1):
              the served token equals the verified tape's argmax there (prefills of 2,040 to 2,111
              tokens: on both sides of the 2,048-key indexer budget and of the prefill chunk)
  concurrent  c = 1..N requests at once (the cases below, a 2,104-token prefill among them):
              every answer's ids equal its solo ids
  stream      one case streamed: the deltas concatenate to the solo answer
  efforts     no_think / high / max / low / medium / xhigh are served (high and max with reasoning,
              no_think without); an unknown level is a 400
  tools       a tool call round trip: the call parses to `get_weather` with JSON arguments, the
              follow-up turn with the tool's answer is served
  sampled     a seeded sampled request is the same answer twice solo and once beside others
  stops       a stop string cuts the answer before its first occurrence; max_tokens cuts the budget
  penalties   a penalized greedy request is served and repeats
  refusals    an image part and a response_format are refused by name
  cancel      a streamed request whose client leaves after three tokens ends at the next step
              (its session freed: the server's step log shows it gone) and the route serves on
  admission   three requests that each book 40% of the served context: the memory book admits
              two at a time and queues the third (no request fails, none runs out of memory)
  long        a needle in a long prompt (about 32k tokens, past the indexer budget sixteen times
              over) is found, greedy, reasoning off
Writes every response and verdict to OUT_JSON; exits 1 when a gate fails."""
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request

addr, key, model, req_dir, tapes_json, gen_json, out_json, server_log = sys.argv[1:9]
QUICK = "--quick" in sys.argv[9:]
EOS = 120025
REQ = {n: json.load(open(f"{req_dir}/{n}.json")) for n in ("chat", "long", "long-split")}
TAPES = json.load(open(tapes_json))
GEN = json.load(open(gen_json))


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


report = {"gates": {}, "cases": {}}
ok_all = True


def verdict(name, ok, detail):
    global ok_all
    report["gates"][name] = {"pass": bool(ok), "detail": detail}
    ok_all &= bool(ok)
    print(f"{name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)


def complete(ids, max_tokens, **kw):
    body = {"model": model, "prompt_ids": ids, "max_tokens": max_tokens, "temperature": 0}
    body.update(kw)
    st, raw, dt = post("/v1/completions", body)
    if st != 200:
        return {"status": st, "error": raw, "wall_s": round(dt, 3)}
    j = json.loads(raw)
    return {"status": st, "tokens": j.get("tokens"), "text": j.get("text"),
            "stop": j.get("stop_reason"), "n_tokens": j.get("n_tokens"), "wall_s": round(dt, 3)}


def detok(ids):
    st, raw, _ = post("/v1/detokenize", {"model": model, "tokens": ids})
    return json.loads(raw)["prompt"] if st == 200 else None


def tokenize(messages, kwargs=None, tools=None):
    body = {"model": model, "messages": messages}
    if kwargs is not None:
        body["chat_template_kwargs"] = kwargs
    if tools is not None:
        body["tools"] = tools
    st, raw, _ = post("/v1/tokenize", body)
    return json.loads(raw).get("tokens") if st == 200 else None


def chat(messages, **kw):
    body = {"model": model, "messages": messages, "max_tokens": 200, "temperature": 0}
    body.update(kw)
    st, raw, dt = post("/v1/chat/completions", body)
    if st != 200:
        return {"status": st, "error": raw, "wall_s": round(dt, 3)}
    j = json.loads(raw)
    ch = j["choices"][0]
    return {"status": st, "content": ch["message"].get("content"),
            "reasoning": ch["message"].get("reasoning"),
            "tool_calls": ch["message"].get("tool_calls"), "finish": ch.get("finish_reason"),
            "usage": j.get("usage"), "wall_s": round(dt, 3)}


def upto_eos(ids):
    return ids[: ids.index(EOS)] if EOS in ids else ids


# ---- render: the probes' user turns, recovered from their own prompt text ----
render_ok = True
for name in ("chat", "long"):
    r = REQ[name]
    prompt = r["token_ids"][: r["decode_from"]]
    text = detok(prompt)
    start = "\uff5chy_start:opensource\uff5c>user<\uff5chy_middle:opensource\uff5c>"
    end = "<\uff5chy_end:opensource\uff5c>"
    a = text.index(start) + len(start)
    user = text[a: text.index(end, a)]
    got = tokenize([{"role": "user", "content": user}], {"enable_thinking": False}) or []
    # the rendered prompt ends at the generation prompt; `long`'s prefill also holds the first
    # tokens of its forced answer (gen_probes.py), which follow the render
    tail = prompt[len(got):]
    same = len(got) > 0 and got == prompt[: len(got)] and (
        not tail or detok(tail).startswith("It describes"))
    render_ok &= same
    report["cases"].setdefault(name, {})["render"] = {"n": len(prompt), "equal": same,
                                                      "rendered": len(got), "answer_tail": len(tail)}
verdict("render", render_ok, "the chat and long probes through /v1/tokenize (reasoning off) equal the request ids up to the generation prompt")

# ---- route: served greedy vs the route's own greedy ----
route_ok = True
for name in ("chat", "long", "long-split"):
    r = REQ[name]
    prompt = r["token_ids"][: r["decode_from"]]
    want = upto_eos(GEN[name])
    got = complete(prompt, len(GEN[name]))
    toks = got.get("tokens") or []
    same = toks == want and got.get("stop") == ("Eos" if EOS in GEN[name] else "MaxNew")
    first = toks[:1] == TAPES[name][r["decode_from"] - 1: r["decode_from"]]
    route_ok &= same and first
    report["cases"].setdefault(name, {})["route"] = got | {"want": want, "equal_route": same,
                                                           "first_equals_tape": first}
verdict("route", route_ok, "served greedy ids equal hy4f-gpu --gen on chat, long, long-split; the first equals the verified tape's argmax")

# ---- forced: teacher-forced positions as prompts ----
forced_ok, n_forced = True, 0
for name in ("chat", "long", "long-split"):
    r = REQ[name]
    ids, d = r["token_ids"], r["decode_from"]
    positions = list(range(d + 1, len(ids) + 1))
    if name == "long-split":
        # the budget crossing and every 8th position after it (the quick run takes fewer)
        positions = sorted({p for p in positions if p in (2047, 2048, 2049, 2050) or p % 8 == 0})
    if QUICK:
        positions = positions[:2]
    rows = []
    for p in positions:
        got = complete(ids[:p], 1)
        want = TAPES[name][p - 1]
        t = (got.get("tokens") or [None])[0]
        if t is None and got.get("stop") == "Eos":
            t = EOS
        rows.append({"prompt": p, "got": t, "want": want, "wall_s": got.get("wall_s")})
        forced_ok &= t == want
        n_forced += 1
    report["cases"][name]["forced"] = rows
verdict("forced", forced_ok, f"{n_forced} teacher-forced positions served as prompts equal the verified tapes' argmax")

# ---- the cases for concurrency, stream, sampling ----
CASES = {
    "capital": ([{"role": "user", "content": "What is the capital of France? Answer in one word."}], False),
    "math": ([{"role": "user", "content": "What is 17 * 23? Answer with the number."}], True),
    "poem": ([{"role": "user", "content": "Write a four-line poem about the sea."}], False),
    "json": ([{"role": "user", "content": "Give the chemical symbols of iron, gold and silver as a JSON object."}], False),
    "hebrew": ([{"role": "user", "content": "\u05de\u05d4 \u05d1\u05d9\u05e8\u05ea \u05e6\u05e8\u05e4\u05ea? \u05e2\u05e0\u05d4 \u05d1\u05de\u05e9\u05e4\u05d8 \u05d0\u05d7\u05d3."}], False),
}
case_ids = {}
for cid, (msgs, think) in CASES.items():
    case_ids[cid] = tokenize(msgs, None if think else {"enable_thinking": False})
case_ids["longdoc"] = REQ["long"]["token_ids"][: REQ["long"]["decode_from"]]
ORDER = ["longdoc", "capital", "math", "poem", "json", "hebrew"]
MAXT = 48
solo = {cid: complete(case_ids[cid], MAXT) for cid in ORDER}
report["solo"] = solo
conc_ok = True
levels = [2, 3, 4, 6] if not QUICK else [2, 4]
for c in levels:
    picked = (ORDER * 2)[:c] if c <= len(ORDER) else ORDER + ORDER[: c - len(ORDER)]
    with concurrent.futures.ThreadPoolExecutor(c) as ex:
        futs = [ex.submit(complete, case_ids[cid], MAXT) for cid in picked]
        outs = [f.result() for f in futs]
    same = all(o.get("tokens") == solo[cid].get("tokens") and o.get("status") == 200
               for cid, o in zip(picked, outs))
    conc_ok &= same
    report.setdefault("concurrent", {})[str(c)] = {"cases": picked, "equal_solo": same,
                                                   "outs": outs}
verdict("concurrent", conc_ok, f"c = {levels} at once (a 2,104-token prefill among them): ids equal solo")

# ---- stream ----
cid = "poem"
msgs = CASES[cid][0]
sreq = urllib.request.Request(
    f"http://{addr}/v1/chat/completions",
    data=json.dumps({"model": model, "messages": msgs, "max_tokens": MAXT, "temperature": 0,
                     "reasoning_effort": "none", "stream": True}).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)
deltas, finish, done_seen, s_err = [], None, False, None
with urllib.request.urlopen(sreq, timeout=3600) as r:
    if r.status != 200:
        s_err = f"HTTP {r.status}"
    for line in r:
        line = line.decode().strip()
        if line == "data: [DONE]":
            done_seen = True
            continue
        if not line.startswith("data: "):
            continue
        j = json.loads(line[6:])
        if "error" in j:
            s_err = json.dumps(j["error"])
        for ch in j.get("choices", []):
            d = (ch.get("delta") or {}).get("content")
            if d:
                deltas.append(d)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
blocking = chat(msgs, max_tokens=MAXT, reasoning_effort="none")
streamed = "".join(deltas)
report["stream"] = {"content": streamed, "chunks": len(deltas), "finish": finish,
                    "done": done_seen, "error": s_err, "blocking": blocking}
verdict("stream", s_err is None and done_seen and finish == blocking.get("finish")
        and blocking.get("status") == 200 and streamed == blocking.get("content") and streamed != "",
        f"{len(deltas)} deltas of {cid} equal the blocking answer")

# ---- efforts ----
eff = {}
for level in ["no_think", "none", "high", "max", "xhigh", "low", "medium"]:
    eff[level] = chat(CASES["math"][0], max_tokens=600, reasoning_effort=level)
bad = chat(CASES["math"][0], max_tokens=8, reasoning_effort="turbo")
report["efforts"] = eff | {"turbo": bad}
e_ok = all(eff[k]["status"] == 200 for k in eff) and bad["status"] == 400
e_ok &= not eff["no_think"].get("reasoning") and not eff["none"].get("reasoning")
e_ok &= bool(eff["high"].get("reasoning")) and bool(eff["max"].get("reasoning"))
# low and medium render the high rung, xhigh the max rung: the same prompt, the same answer
e_ok &= eff["low"].get("reasoning") == eff["high"].get("reasoning") == eff["medium"].get("reasoning")
e_ok &= eff["xhigh"].get("reasoning") == eff["max"].get("reasoning")
e_ok &= eff["no_think"].get("content") == eff["none"].get("content")
verdict("efforts", e_ok, "no_think/none without reasoning; high, max with; low/medium = high; xhigh = max; turbo a 400")

# ---- tools: a round trip ----
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
tmsgs = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
t1 = chat(tmsgs, tools=tools, max_tokens=400, reasoning_effort="none")
calls = t1.get("tool_calls") or []
call_ok = (t1.get("finish") == "tool_calls" and len(calls) >= 1
           and calls[0]["function"]["name"] == "get_weather"
           and isinstance(json.loads(calls[0]["function"]["arguments"]), dict))
t2 = None
if call_ok:
    follow = tmsgs + [
        {"role": "assistant", "content": t1.get("content") or "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": "{\"temp_c\": 18, \"sky\": \"clear\"}"},
    ]
    t2 = chat(follow, tools=tools, max_tokens=200, reasoning_effort="none")
think_t = chat(tmsgs, tools=tools, max_tokens=1200, reasoning_effort="high")
report["tools"] = {"call": t1, "follow_up": t2, "with_reasoning": think_t}
tools_ok = call_ok and t2 is not None and t2["status"] == 200 and bool(t2.get("content")) \
    and not t2.get("tool_calls")
tools_ok &= think_t.get("status") == 200 and (think_t.get("finish") == "tool_calls") \
    and bool(think_t.get("reasoning"))
verdict("tools", tools_ok, "the call parses (get_weather, JSON arguments) with reasoning off and on; the follow-up with the tool's answer is served")

# ---- sampled, seeded ----
sb = dict(temperature=0.8, top_p=0.95, top_k=40, seed=1017, max_tokens=48)
s1 = complete(case_ids["poem"], **sb)
s2 = complete(case_ids["poem"], **sb)
with concurrent.futures.ThreadPoolExecutor(4) as ex:
    futs = [ex.submit(complete, case_ids["poem"], **sb)] + \
        [ex.submit(complete, case_ids[c], MAXT) for c in ("capital", "json", "hebrew")]
    s3 = futs[0].result()
    peers = [f.result() for f in futs[1:]]
s_ok_peers = all(p.get("status") == 200 and p.get("tokens") for p in peers)
report["sampled"] = {"solo1": s1, "solo2": s2, "beside": s3}
s_ok = all(x.get("status") == 200 and x.get("tokens") for x in (s1, s2, s3))
verdict("sampled", s_ok and s1["tokens"] == s2["tokens"] == s3["tokens"] and s1["tokens"] != solo["poem"].get("tokens"),
        "seed 1017, T 0.8, top_p 0.95, top_k 40: twice solo and once beside three others, unlike greedy")
if not s_ok_peers:
    verdict("sampled-peers", False, "a request beside the sampled one failed")

# ---- stops ----
nt = detok(solo["json"]["tokens"])
stop = nt[len(nt) // 2: len(nt) // 2 + 3]
want_stop = nt[: nt.index(stop)]
a_s = chat(CASES["json"][0], max_tokens=MAXT, reasoning_effort="none", stop=[stop])
a_l = complete(case_ids["json"], 5)
report["stops"] = {"stop_string": stop, "want": want_stop, "got": a_s, "max_tokens_5": a_l}
verdict("stops", a_s.get("content") == want_stop and a_s.get("finish") == "stop"
        and a_l.get("tokens") == solo["json"]["tokens"][:5] and a_l.get("stop") == "MaxNew",
        f"stop {stop!r} cuts json before its first occurrence; max_tokens 5 cuts at the budget")

# ---- penalties ----
pb = dict(frequency_penalty=0.5, presence_penalty=0.3)
p1 = complete(case_ids["poem"], MAXT, **pb)
p2 = complete(case_ids["poem"], MAXT, **pb)
report["penalties"] = {"run1": p1, "run2": p2}
verdict("penalties", p1.get("status") == 200 and p1.get("tokens") == p2.get("tokens"),
        "frequency 0.5, presence 0.3 greedy: served, identical twice")

# ---- refusals ----
img = chat([{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}}]}],
           max_tokens=4)
rf = chat(CASES["json"][0], max_tokens=8, response_format={"type": "json_object"})
report["refusals"] = {"image": img, "response_format": rf}
verdict("refusals", img["status"] == 400 and rf["status"] == 400,
        f"image part -> {img['status']}, response_format -> {rf['status']}")

# ---- cancel: the client leaves mid-stream ----
import http.client
import socket
import threading
cancel_deadline = time.monotonic() + 300
conn = http.client.HTTPConnection(addr.split(":")[0], int(addr.split(":")[1]), timeout=300)
conn.connect()


def _expire():
    # one absolute deadline over headers and body: shut the socket, which ends any blocked read
    try:
        conn.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


watchdog = threading.Timer(300, _expire)
watchdog.start()
conn.request("POST", "/v1/completions", body=json.dumps(
    {"model": model, "prompt_ids": case_ids["poem"], "max_tokens": 400, "temperature": 0,
     "ignore_eos": True, "stream": True}),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
seen, cancel_err = 0, None
try:
    resp = conn.getresponse()
    if resp.status != 200:
        cancel_err = f"HTTP {resp.status}"
    while cancel_err is None and seen < 3:
        try:
            raw_line = resp.fp.readline()
        except (TimeoutError, OSError) as e:
            cancel_err = f"read: {e}"
            break
        if not raw_line:
            cancel_err = "the stream ended before three tokens"
            break
        line = raw_line.decode().strip()
        if line.startswith("data: ") and '"error"' in line:
            cancel_err = line
        elif line.startswith("data: ") and '"text"' in line:
            seen += 1
except (TimeoutError, OSError, http.client.HTTPException) as e:
    cancel_err = f"headers: {e}"
finally:
    watchdog.cancel()
    conn.close()
if cancel_err is None and time.monotonic() > cancel_deadline:
    cancel_err = "the third token came after the 300 s deadline"
time.sleep(5)
after = complete(case_ids["capital"], MAXT)
steps = [l for l in open(server_log, errors="replace") if "[hy4f-step]" in l]
last = steps[-1] if steps else ""
report["cancel"] = {"seen": seen, "error": cancel_err, "after": after, "last_step": last.strip()}
verdict("cancel", cancel_err is None and seen == 3 and after.get("status") == 200
        and after.get("tokens") == solo["capital"].get("tokens") and "running=1 " in last,
        "a client that left after 3 of 400 tokens: the route ran on, the next request alone on it")

# ---- admission: sessions that do not fit together wait ----
mreq = urllib.request.Request(f"http://{addr}/v1/models", headers={"Authorization": f"Bearer {key}"})
with urllib.request.urlopen(mreq, timeout=60) as r:
    models = json.loads(r.read().decode())
ctx = next(m for m in models["data"] if m["id"] == model).get("context_length") or 0
big = int(ctx * 0.4)
# the poem's answer runs to its end-of-sentence id (longer than the gates' 48-token budget), so
# the requests overlap; its greedy answer is known
full_poem = complete(case_ids["poem"], 400)
mark = sum(1 for l in open(server_log, errors="replace") if "[admit-mem]" in l)
with concurrent.futures.ThreadPoolExecutor(3) as ex:
    outs = [f.result() for f in [ex.submit(complete, case_ids["poem"], big) for _ in range(3)]]
lines = [l.strip() for l in open(server_log, errors="replace") if "[admit-mem]" in l][mark:]
admits = [l for l in lines if "verdict=admit" in l]
waits = [l for l in lines if "verdict=wait" in l]
running_at = [int(l.split("running=")[1].split()[0]) for l in admits]
want = full_poem.get("tokens")
report["admission"] = {"context": ctx, "max_tokens": big, "want": full_poem, "outs": outs,
                       "admits": admits, "waits": waits}
verdict("admission", ctx > 0 and full_poem.get("stop") == "Eos" and bool(want)
        and all(o.get("status") == 200 and o.get("tokens") == want and o.get("stop") == "Eos" for o in outs)
        and sorted(running_at) == [0, 1, 1] and len(waits) == 1,
        f"3 x {big}-token budgets on a {ctx}-position context: admitted with {running_at} running, {len(waits)} deferred, each the full greedy answer")

# ---- long: a needle past the indexer budget ----
if not QUICK:
    doc = detok(REQ["long"]["token_ids"][: REQ["long"]["decode_from"]])
    start = "Read the text below, then say in one sentence what it is about.\n\n"
    b0 = doc.index(start) + len(start)
    body_text = doc[b0: doc.index("<\uff5chy_end:opensource\uff5c>", b0)]
    parts = [body_text] * 15
    parts.insert(8, "\n\nThe vault code is 4817. Remember it.\n\n")
    lmsg = [{"role": "user", "content": "".join(parts) + "\n\nWhat is the vault code? Answer with the number only."}]
    lids = tokenize(lmsg, {"enable_thinking": False})
    t0 = time.time()
    la = complete(lids, 16)
    lt = time.time() - t0
    text = detok(la.get("tokens") or []) or ""
    report["long"] = {"prompt_tokens": len(lids), "answer": la, "text": text, "wall_s": lt}
    verdict("long", "4817" in text, f"{len(lids)}-token prompt, answer {text!r} in {lt:.0f}s")

report["pass"] = ok_all
json.dump(report, open(out_json, "w"), ensure_ascii=False, indent=1)
print("CHAT-GATE", "PASS" if ok_all else "FAIL", flush=True)
sys.exit(0 if ok_all else 1)
