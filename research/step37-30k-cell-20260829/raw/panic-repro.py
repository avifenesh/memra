import json, time, os, urllib.request
port = os.environ["P"]
def turn(msgs, label, maxtok=256):
    body = {"model": "step37", "messages": msgs, "stream": True, "max_tokens": maxtok,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first=None; text=[]
    r = urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=3600)
    while True:
        line = r.readline()
        if not line: break
        s = line.decode("utf-8","replace").strip()
        if not s.startswith("data:") or "[DONE]" in s: continue
        try: j = json.loads(s[5:].strip())
        except Exception: continue
        d = (j.get("choices") or [{}])[0].get("delta") or {}
        frag = (d.get("reasoning") or "") + (d.get("content") or "")
        if frag:
            if first is None: first = time.perf_counter()-t0
            text.append(frag)
    r.close()
    print("%s ttft=%s len=%d" % (label, first, len("".join(text))), flush=True)
    return "".join(text)
doc = json.load(open("/root/curve-30k-1.json"))
msgs = list(doc["messages"])
reply = turn(msgs, "cold")
msgs.append({"role":"assistant","content":reply})
msgs.append({"role":"user","content":"Now give only item (1) from your instructions, as a numbered list, nothing else."})
try: turn(msgs, "warm")
except Exception as e: print("warm FAIL %s" % type(e).__name__, flush=True)
