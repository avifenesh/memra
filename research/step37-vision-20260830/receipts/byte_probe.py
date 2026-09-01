import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
payload = {
    "model": "step37",
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain in two sentences why the sky is blue."},
    ],
    "max_tokens": 400,
    "temperature": 0,
}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
body = json.load(urllib.request.urlopen(req, timeout=1800))
c = body["choices"][0]
rec = {"content": c["message"].get("content"),
       "reasoning": c["message"].get("reasoning_content"),
       "finish": c.get("finish_reason"), "usage": body.get("usage")}
open(out, "w").write(json.dumps(rec, sort_keys=True))
print(out, "written", rec["finish"], body.get("usage", {}).get("completion_tokens"))
