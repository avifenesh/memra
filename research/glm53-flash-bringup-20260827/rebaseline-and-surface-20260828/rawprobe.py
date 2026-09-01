#!/usr/bin/env python3
"""Raw /v1/completions probe: NO chat template involved. The attribution discriminator.
usage: rawprobe.py <label> <mode: file|pool> <arg> <max_tokens> [greedy|sampled]
  file <path>   -> POST the file bytes verbatim as `prompt`
  pool <idx>    -> POST prompts.json["decode"][idx]["text"] verbatim as `prompt`
"""
import hashlib, json, sys, time, urllib.request

LABEL, MODE, ARG, MT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ARM = sys.argv[5] if len(sys.argv) > 5 else "greedy"
if MODE == "file":
    prompt = open(ARG, encoding="utf-8").read()
else:
    prompt = json.load(open("/home/ubuntu/prompts.json"))["decode"][int(ARG)]["text"]

body = {"model": "zai/glm-5.3-flash", "prompt": prompt, "max_tokens": MT, "stream": False}
if ARM == "greedy":
    body["temperature"] = 0.0

t0 = time.time(); err = None; r = None
req = urllib.request.Request("http://127.0.0.1:18400/v1/completions",
                             data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=1800) as resp:
        r = json.loads(resp.read().decode())
except Exception as e:
    body_txt = ""
    try:
        body_txt = e.read().decode()[:400]
    except Exception:
        pass
    err = f"{type(e).__name__}: {e} {body_txt}"[:500]
t1 = time.time()
ch = ((r or {}).get("choices") or [{}])[0]
txt = ch.get("text", "") or ""
u = (r or {}).get("usage") or {}
print(json.dumps({
    "label": LABEL, "mode": MODE, "arg": ARG, "arm": ARM, "max_tokens": MT,
    "prompt_sha16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
    "prompt_chars": len(prompt),
    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
    "server_elapsed_s": u.get("elapsed_s"),
    "server_toks": (round(u["completion_tokens"]/u["elapsed_s"], 4)
                    if u.get("elapsed_s") else None),
    "finish_reason": ch.get("finish_reason"),
    "wall_s": round(t1-t0, 4),
    "out_sha16": hashlib.sha256(txt.encode()).hexdigest()[:16],
    "out_len": len(txt), "head": txt[:200], "error": err,
}))
