import json, time, os, urllib.request

port = os.environ["P"]; arm = os.environ["ARM"]; rnd = os.environ["RND"]
out = open("/root/long30k-rows.txt", "a", buffering=1)

def stream_turn(msgs, label, maxtok=256):
    body = {"model": "step37", "messages": msgs, "stream": True, "max_tokens": maxtok,
            "stream_options": {"include_usage": True}}   # vendor-default sampled: no temp/top_p
    t0 = time.perf_counter(); first = None; text = []; usage = None
    r = urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}),
        timeout=3600)
    while True:
        line = r.readline()
        if not line: break
        s = line.decode("utf-8", "replace").strip()
        if not s.startswith("data:") or "[DONE]" in s: continue
        try: j = json.loads(s[5:].strip())
        except Exception: continue
        if j.get("usage"): usage = j["usage"]
        d = (j.get("choices") or [{}])[0].get("delta") or {}
        frag = (d.get("reasoning") or "") + (d.get("content") or "")
        if frag:
            if first is None: first = time.perf_counter() - t0
            text.append(frag)
    r.close()
    tot = time.perf_counter() - t0
    reply = "".join(text)
    sp = (usage or {}).get("spec") or {}
    n = len(text)
    tps = (n - 1) / (tot - first) if (first is not None and n > 1) else -1
    if first is None or not reply:
        out.write("rnd=%s arm=%s leg=%s INVALID empty\n" % (rnd, arm, label)); return None
    out.write("rnd=%s arm=%s leg=%s ttft=%.3f tok_s=%.1f prompt=%s acc=%s rounds=%s len=%d\n" % (
        rnd, arm, label, first, tps, (usage or {}).get("prompt_tokens"),
        sp.get("acceptance_rate"), sp.get("rounds"), len(reply)))
    return reply

for i in (1, 2):
    doc = json.load(open("/root/curve-30k-%d.json" % i))
    msgs = list(doc["messages"])
    try:
        reply = stream_turn(msgs, "cold30k-%d" % i)      # LEG 1: cold 30k+ prime
        if reply is None: continue
        msgs.append({"role": "assistant", "content": reply})
        msgs.append({"role": "user", "content": "Now give only item (1) from your instructions, as a numbered list, nothing else."})
        stream_turn(msgs, "warm30k-%d" % i)              # LEG 2: short suffix on the 30k session
    except Exception as e:
        out.write("rnd=%s arm=%s doc=%d FAIL %s\n" % (rnd, arm, i, type(e).__name__))
out.close()
