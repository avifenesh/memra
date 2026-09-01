#!/usr/bin/env python3
"""Where does a MONOLITHIC prime actually stop being servable, once the tail ring no longer
caps it?

The ring fix removes the ring as the ceiling. The next ceiling up is arithmetic, not a flag:
`mla_kpool_indices` allocates its score plane as `t * n_pools` f32 with `n_pools = t_kv / pool`
and `pool = 4`, so under a monolithic prime (t = t_kv = N) that plane is N**2/4 floats, i.e.
EXACTLY N**2 BYTES, per MLA layer, per call. Predicted: 67 MB at 8k, 1.07 GB at 32k, 4.3 GB at
64k, 17.2 GB at 128k. The index list adds a linear 4 * N * 2051 bytes on top.

This probe does not fix that and does not investigate it. It brackets it, because the number is
the honest upper bound on any context claim for this model.

Run with the server booted at a MEMRA_CTX large enough that these prompts FIT the configured
window, so a refusal is about capacity and not about admission.

usage: wallprobe.py <label> <outfile> [targets...]
"""
import json, sys, time, urllib.request, urllib.error

LABEL, OUTFILE = sys.argv[1], sys.argv[2]
TARGETS = [int(a) for a in sys.argv[3:]] or [8000, 16000, 32000, 64000, 128000]
EP = "http://127.0.0.1:18400"
MODEL = "zai/glm-5.3-flash"
UNIT = "The quick brown fox jumps over the lazy dog. "

NEEDLES = [("tail ring lapped", "RING-LAPPED"),
           ("out_of_memory", "CUDA-OOM"),
           ("out of memory", "CUDA-OOM"),
           ("invalid_value", "CUDA-INVALID-VALUE"),
           ("invalid argument", "CUDA-INVALID-VALUE"),
           ("max_ctx", "ADMISSION"),
           ("exceeds", "ADMISSION")]


def classify(err):
    e = (err or "").lower()
    for needle, name in NEEDLES:
        if needle in e:
            return name
    return "OTHER" if e else "-"


rows = []
for target in TARGETS:
    reps = max(1, int(target * 4.15 / len(UNIT)))
    # STREAMED: the 90s deadline is a platform ceiling on nonstream responses; the streaming
    # escape is the documented path for longer work, so the ladder rides it. ttfd is the
    # prefill wall on this eager model.
    body = {"model": MODEL,
            "messages": [{"role": "user",
                          "content": UNIT * reps + "\n\nReply with the single word: ok"}],
            "max_tokens": 8, "temperature": 0.0, "reasoning_effort": "low",
            "stream": True}
    req = urllib.request.Request(EP + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    first = None
    pt = None
    err = ""
    st = -1
    try:
        with urllib.request.urlopen(req, timeout=2400) as r:
            st = r.status
            for line in r:
                if line.startswith(b"data:"):
                    if first is None:
                        first = time.time() - t0
                    frag = line[5:].strip()
                    if frag and frag != b"[DONE]":
                        try:
                            d = json.loads(frag)
                            u = d.get("usage") or {}
                            pt = u.get("prompt_tokens") or pt
                            e2 = (d.get("error") or {}).get("message")
                            if e2:
                                err = e2
                        except Exception:
                            pass
    except urllib.error.HTTPError as e:
        st = e.code
        err = e.read().decode()[:300]
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    wall = time.time() - t0
    n = pt or target
    cls = classify(err)
    ok = st == 200 and not err
    row = {"label": LABEL, "target_tokens": target, "status": st, "prompt_tokens": pt,
           "class": cls, "wall_s": round(wall, 1),
           "ttfd_s": round(first, 1) if first is not None else None,
           "error": err[:300] or None}
    rows.append(row)
    tail = "OK" if ok else repr(err[:120])
    print("  ~%7d tok: status=%4s prompt_tokens=%7s ttfd=%8s wall=%7.1fs class=%-18s %s"
          % (target, st, pt, ("%.1fs" % first) if first is not None else "none", wall, cls, tail),
          flush=True)
    json.dump(rows, open(OUTFILE, "w"), indent=1)

ok = [r["prompt_tokens"] or r["target_tokens"] for r in rows if r["status"] == 200 and not r["error"]]
bad = [r for r in rows if r["status"] != 200 or r["error"]]
print("  == %s: largest SERVED = %s tokens; first failure at ~%s (%s)"
      % (LABEL, max(ok) if ok else None,
         bad[0]["target_tokens"] if bad else "n/a",
         bad[0]["class"] if bad else "-"))
