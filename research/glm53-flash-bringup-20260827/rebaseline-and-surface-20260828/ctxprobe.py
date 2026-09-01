#!/usr/bin/env python3
"""Is the DSA index tail-ring (f7ec4f557b, MEMRA_DSA_INDEX_RING default ON) safe INSIDE the
served context window?

The follow-up cell only probed sizes ABOVE MEMRA_CTX=8192, where a refusal of some kind is
expected anyway. The question that decides whether f7ec4f557b is a regression is narrower:
does a prompt that FITS the configured window (<= 8192 tokens) serve?

The ring's own guard is `pools_ready*pool + ring >= slot + t`, and it failed at ~13k tokens with
"indexer tail ring lapped: 5120 rows cannot cover pools from row 0". 5120 rows is the ring size,
so the boundary should sit near 5120 tokens, i.e. INSIDE the 8192-token window.

Arms are the same binary with the flag's documented rollback seam:
  ring ON  (default)          MEMRA_DSA_INDEX_RING unset
  ring OFF (rollback)         MEMRA_DSA_INDEX_RING=0
plus the pre-f7ec binary, which has no ring at all.

usage: ctxprobe.py <label> <outfile>
"""
import json, sys, urllib.request, urllib.error

LABEL, OUTFILE = sys.argv[1], sys.argv[2]
EP = "http://127.0.0.1:18400"
MODEL = "zai/glm-5.3-flash"

# "The quick brown fox jumps over the lazy dog. " measured ~4.15 chars/token on this tokenizer.
UNIT = "The quick brown fox jumps over the lazy dog. "
TARGETS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 7900]

rows = []
for target in TARGETS:
    reps = max(1, int(target * 4.15 / len(UNIT)))
    body = {"model": MODEL,
            "messages": [{"role": "user",
                          "content": UNIT * reps + "\n\nReply with the single word: ok"}],
            "max_tokens": 8, "temperature": 0.0, "reasoning_effort": "low"}
    req = urllib.request.Request(EP + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            raw = r.read().decode(); st = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode(); st = e.code
    except Exception as e:
        raw = f"{type(e).__name__}: {e}"; st = -1
    try:
        j = json.loads(raw)
    except Exception:
        j = {}
    u = (j.get("usage") or {}) if isinstance(j, dict) else {}
    err = ((j.get("error") or {}).get("message") if isinstance(j, dict) else None) or ""
    ch = ((j.get("choices") or [{}])[0]) if isinstance(j, dict) else {}
    row = {"label": LABEL, "target_tokens": target, "chars": len(body["messages"][0]["content"]),
           "status": st, "prompt_tokens": u.get("prompt_tokens"),
           "content": ((ch.get("message") or {}).get("content") or "")[:40],
           "error": err[:220] or None}
    rows.append(row)
    print(f"  ~{target:>5} tok ({row['chars']:>7} chars): status={st:>4} "
          f"prompt_tokens={str(u.get('prompt_tokens')):>6} "
          f"{'OK ' + repr(row['content']) if st == 200 else 'ERR ' + repr(err[:110])}",
          flush=True)

json.dump(rows, open(OUTFILE, "w"), indent=1)
served = [r["prompt_tokens"] for r in rows if r["status"] == 200 and r["prompt_tokens"]]
failed = [(r["prompt_tokens"] or r["target_tokens"], r["error"]) for r in rows if r["status"] != 200]
print(f"  == {LABEL}: largest SERVED prompt = {max(served) if served else None} tokens; "
      f"{len(failed)} failure(s) at/above ~{failed[0][0] if failed else 'n/a'} tokens")
