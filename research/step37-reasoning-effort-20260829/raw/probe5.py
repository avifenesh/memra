import json, sys, urllib.request, urllib.error

port, out_path, tag = sys.argv[1], sys.argv[2], sys.argv[3]
CHAT = "http://127.0.0.1:%s/v1/chat/completions" % port
PROMPTS = [
 ("mul",   "What is 17*23? Reply with the number only.", "391"),
 ("cap",   "What is the capital of France? Reply with one word.", "Paris"),
 ("count", "How many days are in a leap year? Reply with the number only.", "366"),
 ("word",  "Reverse the word: stressed. Reply with one word.", "desserts"),
]
def post(payload):
    r = urllib.request.Request(CHAT, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(r, timeout=900)
        return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        try: b = json.loads(e.read().decode() or "{}")
        except Exception: b = {}
        return e.code, b

rows = []
for name, q, want in PROMPTS:
    for rep in range(2):
        code, body = post({"model":"step37","temperature":0.0,"max_tokens":320,
                           "messages":[{"role":"user","content":q}]})
        row = {"tag": f"{tag}/{name}/r{rep}", "http": code, "want": want}
        if code == 200:
            ch = body["choices"][0]; m = ch.get("message") or {}
            row.update(reasoning=m.get("reasoning") or "", content=m.get("content") or "",
                       usage=body.get("usage"))
            hit = want.lower() in ((row["content"] or "")+(row["reasoning"] or "")).lower()
            pt = (body.get("usage") or {}).get("prompt_tokens")
            print(f"{row[chr(116)+chr(97)+chr(103)]}: pt={pt} HIT={hit} R={(row[chr(114)+chr(101)+chr(97)+chr(115)+chr(111)+chr(110)+chr(105)+chr(110)+chr(103)])[:55]!r} C={(row[chr(99)+chr(111)+chr(110)+chr(116)+chr(101)+chr(110)+chr(116)])[:35]!r}", flush=True)
        else:
            row["error"] = body
            print(f"{row[chr(116)+chr(97)+chr(103)]}: HTTP {code} {json.dumps(body)[:150]}", flush=True)
        rows.append(row)
json.dump(rows, open(out_path,"w"), indent=1)
print("WROTE", out_path)
