#!/usr/bin/env python3
"""8-draw first-token census on a LIVE server (the near-tie adjudication shape from the
MEMRA_PP_BF16 row). Run once per arm against a fresh boot of that arm.

usage: census8.py <outdir> <arm-label> [draws=8]
Draws greedy C6470 (the flipped prompt), max_tokens=16, streamed; banks each stream's
first chunk + first 16-token text + sha16. A stable argmax difference shows 8/8 the same
text per arm; a near-tie shows within-arm variation.
"""
import hashlib
import json
import sys
import time
import urllib.request

OUT, ARM = sys.argv[1], sys.argv[2]
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
EP = "http://127.0.0.1:18400"
POOL = json.load(open("/root/l3-ab/prompts.json"))
rows = []
for i in range(DRAWS):
    body = {"model": "zai/glm-5.3-flash", "prompt": POOL["C6470"], "max_tokens": 16,
            "stream": True, "temperature": 0.0}
    req = urllib.request.Request(EP + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    chunks, err = [], None
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for rl in r:
                s = rl.decode().strip()
                if not s.startswith("data:"):
                    continue
                p = s[5:].strip()
                if p == "[DONE]":
                    break
                o = json.loads(p)
                c = (o.get("choices") or [{}])[0]
                if c.get("text"):
                    chunks.append(c["text"])
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    text = "".join(chunks)
    rows.append({"arm": ARM, "draw": i, "first_chunk": text[:24],
                 "text16": text, "sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
                 "wall_s": round(time.time() - t0, 3), "error": err})
    print(f"  {ARM} draw{i}: sha={rows[-1]['sha16']} text={text[:48]!r} err={err}",
          flush=True)
uniq = sorted({r["sha16"] for r in rows})
print(f"# census {ARM}: {len(uniq)} unique sha(s) across {DRAWS} draws: {uniq}", flush=True)
json.dump(rows, open(f"{OUT}/census8-{ARM}.json", "w"), indent=1)
