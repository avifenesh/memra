#!/usr/bin/env python3
"""client1307.py OUT.jsonl MODE TAG: the requests of one memra#1307 cell against 127.0.0.1:8090,
greedy (temperature 0, seed 1307), non-streaming chat completions, each with its own nonce line so
no request reuses another's prefix.

- mtp: three requests one after another (m1, m2, m3), a 1k-token document each, 128 tokens: each
  is a fresh session whose draft-graph capture opens and closes one draft-state bracket.
- dflash: two pairs (p1a+p1b, then p2a+p2b), the two requests of a pair sent together, a 2k-token
  document each, 384 tokens: the pair decodes as one DFlash2 cohort, whose pooled verify width
  the change under test decides.
- dflash4: two groups of four sent together (q1a..q1d, then q2a..q2d), the same document size and
  384 tokens: a cohort of four, where the members beyond the first need the most headroom.
- dflash4long: one group of four (s1a..s1d) asked to copy the records, up to 3,072 tokens each:
  a cohort that decodes for minutes, for the short-memory cell.

Each row: request, HTTP status, prompt and completion tokens, finish reason, wall seconds, the
text's sha256 (the arms' identity check) and its length in characters. The text is the message's
reasoning (`reasoning`, or `reasoning_content`), a NUL, and its content. A 200 row with completion
tokens and no text is an error row ("empty text"): the first campaign's client read only
`reasoning_content`, which this server does not send in a non-streaming message, so every hash
was the hash of the lone NUL and compared nothing."""
import hashlib
import json
import sys
import threading
import time
import urllib.request

out, mode, tag = sys.argv[1], sys.argv[2], sys.argv[3]
URL = "http://127.0.0.1:8090/v1/chat/completions"


QUESTION = ("Which station logged the most units in the records above, and at what hour? "
            "Answer in two sentences, then list three records with their units.")
COPY = "Copy every record above exactly, one per line, in order, then answer: which station logged the most units?"


def document(lines, nonce, closing):
    rows = [f"memra-1307 nonce {nonce}"]
    for i in range(lines):
        rows.append(f"Record {i}: station {i * 37 % 101} logged {i * 7919 % 10007} units at hour {i % 24}.")
    rows.append(closing)
    return "\n".join(rows)


def ask(name, lines, max_tokens, closing=QUESTION):
    body = {"model": "q38", "messages": [{"role": "user", "content": document(lines, name, closing)}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 1307}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as f:
            status, payload = f.status, json.load(f)
    except urllib.error.HTTPError as e:
        status, payload = e.code, {"raw": e.read().decode(errors="replace")[:400]}
    except Exception as e:  # noqa: BLE001 - the row records the failure
        status, payload = 0, {"raw": repr(e)[:400]}
    wall = time.time() - t0
    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    text = reasoning + "\x00" + content
    usage = payload.get("usage") or {}
    error = None if status == 200 else payload.get("raw")
    if status == 200 and usage.get("completion_tokens") and not (reasoning or content):
        error = "empty text"
    return {"tag": tag, "req": name, "status": status, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "finish": choice.get("finish_reason"),
            "wall_s": round(wall, 3), "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text_chars": len(reasoning) + len(content), "error": error}


def together(names, lines, max_tokens, closing=QUESTION):
    got = {}
    ts = [threading.Thread(target=lambda n=n: got.update({n: ask(n, lines, max_tokens, closing)})) for n in names]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return [got[n] for n in names]


rows = []
if mode == "mtp":
    for name in ("m1", "m2", "m3"):
        rows.append(ask(name, 80, 128))
elif mode == "dflash":
    for pair in (("p1a", "p1b"), ("p2a", "p2b")):
        rows.extend(together(pair, 160, 384))
elif mode == "dflash4":
    for g in ("q1", "q2"):
        rows.extend(together([g + x for x in "abcd"], 160, 384))
elif mode == "dflash4long":
    rows.extend(together(["s1" + x for x in "abcd"], 160, 3072, COPY))
else:
    sys.exit(f"mode {mode} unknown")
with open(out, "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
        print(json.dumps(r)[:280])
