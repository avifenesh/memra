#!/usr/bin/env python3
"""The deepseek-v4.1 route's served-answer gates against a running memra-server.

usage: chat_gate.py ADDR API_KEY MODEL CASES_JSON DEC_DIR TAPE_IDS_DIR OUT_JSON

For the chat tapes capital, gsm, multiturn, sys_json, hebrew (chat mode, the tapes' thinking_mode), and the
long needle tapes (3,761 and 18,343 prompt tokens) for render and solo:
  render    /v1/tokenize of the case's messages equals the tape's prompt ids
  solo      greedy /v1/chat/completions one request at a time: the content equals the native greedy
            generation (`dsv41-run --gen`, DEC_DIR/greedy/<case>/gen.i32) detokenized, the completion token
            count equals it, finish_reason is stop
  batched   the five requests at once (one runner forward carries all of them): every answer equals its
            solo answer, byte for byte
  stream    one case streamed: the deltas concatenate to the solo answer
  sampled   a seeded sampled request is the same answer twice solo and once beside four others
  budget    reasoning_effort "63" (numeric V4.1 budget) is served; "101" is a 400 naming the range
Writes every response and verdict to OUT_JSON; exits 1 when a gate fails."""
import concurrent.futures
import json
import struct
import sys
import time
import urllib.error
import urllib.request

addr, key, model, cases_json, dec_dir, ids_dir, out_json = sys.argv[1:8]
CASES = ["capital", "gsm", "multiturn", "sys_json", "hebrew"]
# past the 128-token window and the indexer's top-512, the second past the candidate pool's 16,384
LONG = ["needle", "needle1200"]
EOS = 1
cases = {c["id"]: c for c in json.load(open(cases_json))}


def post(path, body, stream=False):
    req = urllib.request.Request(
        f"http://{addr}{path}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            raw = r.read().decode()
            return r.status, raw, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), time.time() - t0


def i32(path):
    b = open(path, "rb").read()
    return list(struct.unpack(f"<{len(b) // 4}i", b))


def chat_body(cid, **kw):
    body = {
        "model": model,
        "messages": cases[cid]["messages"],
        "max_tokens": 200,
        "temperature": 0,
        "reasoning_effort": "none",
    }
    body.update(kw)
    return body


report = {"gates": {}, "cases": {}}
ok_all = True


def verdict(name, ok, detail):
    global ok_all
    report["gates"][name] = {"pass": bool(ok), "detail": detail}
    ok_all &= bool(ok)
    print(f"{name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)


# render
render_ok = True
for cid in CASES + LONG:
    st, raw, _ = post("/v1/tokenize", {"model": model, "messages": cases[cid]["messages"],
                                         "chat_template_kwargs": {"enable_thinking": False}})
    got = json.loads(raw).get("tokens") if st == 200 else None
    want = i32(f"{ids_dir}/{cid}.prompt.i32")
    same = got == want
    render_ok &= same
    report["cases"].setdefault(cid, {})["render"] = {"status": st, "n": len(want), "equal": same,
                                                     "got_head": (got or [])[:8]}
verdict("render", render_ok, f"{len(CASES + LONG)} prompts tokenized through /v1/tokenize vs the tape ids")

# native answers
native = {}
for cid in CASES + LONG:
    gen = i32(f"{dec_dir}/greedy/{cid}/gen.i32")
    body = gen[:-1] if gen and gen[-1] == EOS else gen
    st, raw, _ = post("/v1/detokenize", {"model": model, "tokens": body})
    native[cid] = {"ids": gen, "n": len(body), "eos": bool(gen and gen[-1] == EOS),
                   "text": json.loads(raw)["prompt"] if st == 200 else None, "status": st}
    report["cases"][cid]["native"] = native[cid]


def answer(raw):
    j = json.loads(raw)
    ch = j["choices"][0]
    return {"content": ch["message"].get("content"), "reasoning": ch["message"].get("reasoning"),
            "finish": ch.get("finish_reason"), "usage": j.get("usage")}


# solo
solo = {}
solo_ok = True
for cid in CASES + LONG:
    st, raw, dt = post("/v1/chat/completions", chat_body(cid))
    a = answer(raw) if st == 200 else {"error": raw}
    a["status"], a["wall_s"] = st, round(dt, 3)
    solo[cid] = a
    n = native[cid]
    same = st == 200 and a["content"] == n["text"] and a["usage"]["completion_tokens"] == n["n"] \
        and a["usage"]["prompt_tokens"] == len(i32(f"{ids_dir}/{cid}.prompt.i32")) \
        and a["finish"] == ("stop" if n["eos"] else "length")
    solo_ok &= same
    report["cases"][cid]["solo"] = a | {"equal_native": same}
verdict("solo", solo_ok, f"served greedy content and token counts equal dsv41-run --gen on {len(CASES + LONG)} cases")

# batched
with concurrent.futures.ThreadPoolExecutor(len(CASES)) as ex:
    futs = {cid: ex.submit(post, "/v1/chat/completions", chat_body(cid)) for cid in CASES}
    batched = {cid: f.result() for cid, f in futs.items()}
b_ok = True
for cid, (st, raw, dt) in batched.items():
    a = answer(raw) if st == 200 else {"error": raw}
    same = st == 200 and a["content"] == solo[cid].get("content") \
        and a["usage"]["completion_tokens"] == solo[cid]["usage"]["completion_tokens"]
    b_ok &= same
    report["cases"][cid]["batched"] = a | {"status": st, "wall_s": round(dt, 3), "equal_solo": same}
verdict("batched", b_ok, "five concurrent requests answer byte for byte as solo")

# stream
cid = "sys_json"
req = urllib.request.Request(
    f"http://{addr}/v1/chat/completions",
    data=json.dumps(chat_body(cid, stream=True)).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)
deltas = []
with urllib.request.urlopen(req, timeout=3600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        j = json.loads(line[6:])
        for ch in j.get("choices", []):
            d = ch.get("delta", {}).get("content")
            if d:
                deltas.append(d)
streamed = "".join(deltas)
report["cases"][cid]["stream"] = {"content": streamed, "chunks": len(deltas)}
verdict("stream", streamed == solo[cid]["content"], f"{len(deltas)} deltas of {cid}")

# sampled, seeded
sb = dict(temperature=0.8, top_p=0.95, top_k=40, seed=1017, max_tokens=48)
s1 = answer(post("/v1/chat/completions", chat_body("gsm", **sb))[1])
s2 = answer(post("/v1/chat/completions", chat_body("gsm", **sb))[1])
with concurrent.futures.ThreadPoolExecutor(len(CASES)) as ex:
    futs = {c: ex.submit(post, "/v1/chat/completions", chat_body(c, **sb) if c == "gsm" else chat_body(c))
            for c in CASES}
    s3 = answer(futs["gsm"].result()[1])
report["sampled"] = {"solo1": s1, "solo2": s2, "batched": s3}
verdict("sampled", s1["content"] == s2["content"] == s3["content"],
        "seed 1017, T 0.8, top_p 0.95, top_k 40: two solo runs and one beside four others")

# stops: a stop string inside the native answer cuts there (exclusive), max_tokens cuts at the budget
nt = native["sys_json"]["text"]
stop = nt[len(nt) // 2: len(nt) // 2 + 3]
want_stop = nt[:nt.index(stop)]
st_s, raw_s, _ = post("/v1/chat/completions", chat_body("sys_json", stop=[stop]))
a_s = answer(raw_s) if st_s == 200 else {"error": raw_s}
st_d, raw_d, _ = post("/v1/detokenize", {"model": model, "tokens": native["gsm"]["ids"][:5]})
want_len = json.loads(raw_d)["prompt"] if st_d == 200 else None
st_l, raw_l, _ = post("/v1/chat/completions", chat_body("gsm", max_tokens=5))
a_l = answer(raw_l) if st_l == 200 else {"error": raw_l}
report["stops"] = {"stop_string": stop, "want": want_stop, "got": a_s, "max_tokens_5": a_l, "want_5": want_len}
verdict("stops", a_s.get("content") == want_stop and a_s.get("finish") == "stop"
        and a_l.get("content") == want_len and a_l.get("finish") == "length",
        f"stop {stop!r} cuts sys_json at its first occurrence; max_tokens 5 cuts gsm with finish length")

# penalties: a penalized greedy request is served and repeats
pb = dict(frequency_penalty=0.5, presence_penalty=0.3, max_tokens=48)
p1 = answer(post("/v1/chat/completions", chat_body("gsm", **pb))[1])
p2 = answer(post("/v1/chat/completions", chat_body("gsm", **pb))[1])
report["penalties"] = {"run1": p1, "run2": p2}
verdict("penalties", p1["content"] == p2["content"] and p1["content"] is not None,
        "frequency 0.5, presence 0.3 greedy: served, identical twice")

# numeric budget
st63, raw63, _ = post("/v1/chat/completions", chat_body("capital", reasoning_effort="63", max_tokens=24))
st101, raw101, _ = post("/v1/chat/completions", chat_body("capital", reasoning_effort="101", max_tokens=4))
a63 = answer(raw63) if st63 == 200 else {"error": raw63}
report["budget"] = {"63": a63 | {"status": st63}, "101": {"status": st101, "body": raw101}}
verdict("budget", st63 == 200 and st101 == 400 and "[1, 100]" in raw101,
        f"63 -> {st63} (reasoning {len(a63.get('reasoning') or '')} chars), 101 -> {st101}")

# speed (measured, not gated): one longer greedy generation alone on the route
st, raw, dt = post("/v1/chat/completions", chat_body("story", max_tokens=128))
sp = answer(raw) if st == 200 else {"error": raw}
report["speed"] = sp | {"status": st, "wall_s": round(dt, 3)}
print(f"speed: story {st} {sp.get('usage')} in {dt:.1f}s", flush=True)

report["pass"] = ok_all
json.dump(report, open(out_json, "w"), ensure_ascii=False, indent=1)
print("CHAT-GATE", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
