import json, sys, threading, urllib.request
def one(i):
    body = {"model": "hy4", "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper, at least 600 words."}],
            "max_tokens": 256, "seed": 20261121 + i, "temperature": 0.9, "reasoning_effort": "no_think", "ignore_eos": True}
    req = urllib.request.Request("http://127.0.0.1:18393/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer hy4f-batch", "Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=3600).read()
ts = [threading.Thread(target=one, args=(i,)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
print("REQ-DONE")
