#!/usr/bin/env python3
"""client1316.py OUT.jsonl TAG: the requests of one memra#1316 serving cycle against
127.0.0.1:8090: memra#1307's `dflash` cell (client1307.py), two pairs of greedy non-streaming chat
completions, the two requests of a pair sent together, a 160-record document and 384 tokens each.
They are here to make the server the one both occurrences followed (a DFlash2 server that has
served cohorts), not to be compared: each row keeps the status, the token counts and the wall time."""
import json
import sys
import threading
import time
import urllib.request

out, tag = sys.argv[1], sys.argv[2]
URL = "http://127.0.0.1:8090/v1/chat/completions"
QUESTION = ("Which station logged the most units in the records above, and at what hour? "
            "Answer in two sentences, then list three records with their units.")


def document(lines, nonce):
    rows = [f"memra-1316 nonce {tag} {nonce}"]
    for i in range(lines):
        rows.append(f"Record {i}: station {i * 37 % 101} logged {i * 7919 % 10007} units at hour {i % 24}.")
    rows.append(QUESTION)
    return "\n".join(rows)


def ask(name):
    body = {"model": "q38", "messages": [{"role": "user", "content": document(160, name)}],
            "max_tokens": 384, "temperature": 0, "seed": 1316}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as f:
            status, payload = f.status, json.load(f)
    except urllib.error.HTTPError as e:
        status, payload = e.code, {"raw": e.read().decode(errors="replace")[:400]}
    except Exception as e:  # noqa: BLE001 - the row records the failure
        status, payload = 0, {"raw": repr(e)[:400]}
    usage = payload.get("usage") or {}
    return {"tag": tag, "req": name, "status": status, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "wall_s": round(time.time() - t0, 3),
            "error": None if status == 200 else payload.get("raw")}


rows = []
for pair in (("p1a", "p1b"), ("p2a", "p2b")):
    got = {}
    ts = [threading.Thread(target=lambda n=n: got.update({n: ask(n)})) for n in pair]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    rows.extend(got[n] for n in pair)
with open(out, "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(" ".join(f"{r['req']}={r['status']}/{r['completion_tokens']}" for r in rows))
