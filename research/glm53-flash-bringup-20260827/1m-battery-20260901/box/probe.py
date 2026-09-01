#!/usr/bin/env python3
"""One depth rung of the 1M DEPTH RE-PRICE (lane/glm5-1m-battery).

Adapted from the 1m-demo's primeprobe.py (research/glm53-flash-bringup-20260827/
1m-demo-20260829/primeprobe.py) so the prompts, the slicing, the ask, and the token
accounting are BYTE-COMPARABLE with the banked demo rows. Changes vs the demo probe,
each deliberate:
  * port 18400 (this window's port; the demo served on 18500)
  * steady-state decode p50 in addition to the demo's whole-span decode tok/s, so the
    depth curve is read off the same statistic the demo reported ("steady p50")
  * per-token interarrival list kept, so loop-law screening and the steady window are
    reproducible from the receipt alone

Sampling arms (serving law: NEVER serve greedy; every rung carries a vendor-default twin):
  greedy  temperature 0.0            the byte-receipt instrument, and the demo's arm
  vendor  NO sampling params at all  the real traffic shape (models.toml vendor defaults)
reasoning_effort is PINNED "low" on every arm (LAW:reasoning-effort-pinned-in-decode-cells:
omitting it measures think-prose, not the claim shape, and faked a fleet-wide regression).

Token counts come from the server's own usage.prompt_tokens, never estimated.
usage: probe.py <label> <corpus.txt> <chars> <max_tokens> <greedy|vendor> <out.json>
"""
import json
import sys
import time
import urllib.error
import urllib.request

LABEL, CORPUS, CHARS, MAXTOK, MODE, OUTFILE = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
EP = "http://<ip>:18400"
MODEL = "zai/glm-5.3-flash"
STEADY_SKIP = 8   # drop the first 8 arrivals: prefill spill + first-round spec warmup

text = open(CORPUS, encoding="utf-8", errors="strict").read()[:CHARS]
ask = ("\n\n---\nThe text above is a corpus of classic literature. In one short paragraph, "
       "name two of the works it contains and one theme they share.")
body = {"model": MODEL,
        "messages": [{"role": "user", "content": text + ask}],
        "max_tokens": MAXTOK,
        "reasoning_effort": "low",
        "stream": True,
        "stream_options": {"include_usage": True}}
if MODE == "greedy":
    body["temperature"] = 0.0
elif MODE != "vendor":
    sys.exit(f"mode must be greedy|vendor, got {MODE}")

req = urllib.request.Request(EP + "/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"})
t0 = time.time()
arrivals = []
pieces, reasoning_pieces = [], []
usage = None
err = ""
status = -1
t_first = None
t_last = None
try:
    with urllib.request.urlopen(req, timeout=64800) as r:
        status = r.status
        for line in r:
            now = time.time()
            if not line.startswith(b"data:"):
                continue
            frag = line[5:].strip()
            if not frag or frag == b"[DONE]":
                continue
            try:
                d = json.loads(frag)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            e2 = (d.get("error") or {}).get("message")
            if e2:
                err = e2
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                # glm5 thinks: the first decoded tokens arrive on the REASONING channel,
                # so first-token time and cadence must count both (demo probe's rule).
                c = delta.get("content")
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if c or rc:
                    if t_first is None:
                        t_first = now
                    t_last = now
                    arrivals.append(round(now - t0, 5))
                if rc:
                    reasoning_pieces.append(rc)
                if c:
                    pieces.append(c)
except urllib.error.HTTPError as e:
    status = e.code
    err = e.read().decode(errors="replace")[:800]
except Exception as e:
    err = f"{type(e).__name__}: {e}"

wall = time.time() - t0
out_text = "".join(pieces)
reasoning_text = "".join(reasoning_pieces)
pt = (usage or {}).get("prompt_tokens")
ct = (usage or {}).get("completion_tokens")

# TTFD = time to first decoded token = the prefill wall on this eager model.
prefill_s = round(t_first - t0, 3) if t_first else None
prefill_tps = round(pt / prefill_s, 2) if (pt and prefill_s) else None

# span decode tok/s: the demo's statistic, kept for direct comparability
decode_tps = None
if ct and ct > 1 and t_first and t_last and t_last > t_first:
    decode_tps = round((ct - 1) / (t_last - t_first), 2)

# steady-state decode tok/s: median interarrival over arrivals past STEADY_SKIP.
# Chunk arrivals are not exactly one token each, so this is scaled by the measured
# tokens-per-arrival ratio and reported alongside; the span number stays primary.
steady_tps = None
gaps = [round(arrivals[i + 1] - arrivals[i], 5) for i in range(len(arrivals) - 1)]
steady_gaps = gaps[STEADY_SKIP:] if len(gaps) > STEADY_SKIP + 4 else gaps
if steady_gaps:
    s = sorted(steady_gaps)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    if med > 0:
        tok_per_arrival = (ct / len(arrivals)) if (ct and arrivals) else 1.0
        steady_tps = round(tok_per_arrival / med, 2)
# STEADY-P50 IS NOT VALID ON A SPEC ARM, and a broken statistic must never reach a table.
# A speculative round emits its whole accepted run in one burst, so many interarrival gaps
# are ~0 and the MEDIAN gap collapses: the c3a PP3 spec arm reported steady_p50=50393.7 tok/s
# beside a sound span of 27.23. The span statistic ((ct-1)/(t_last-t_first)) is burst-proof
# because it only reads the endpoints. So: keep p50 for plain arms, and NULL it out whenever
# it disagrees with the span by more than 3x, recording why in the receipt itself.
steady_invalid_reason = None
if steady_tps and decode_tps and steady_tps > 3.0 * decode_tps:
    steady_invalid_reason = (
        f"burst artifact: p50 {steady_tps} > 3x span {decode_tps}; speculative bursts make "
        "many interarrival gaps ~0 so the median gap collapses. Use the span number."
    )
    steady_tps = None

summary = {"label": LABEL, "mode": MODE, "corpus": CORPUS, "chars": CHARS,
           "max_tokens": MAXTOK, "status": status, "usage": usage,
           "prefill_s": prefill_s, "prefill_tok_s": prefill_tps,
           "decode_tok_s": decode_tps, "decode_steady_tok_s": steady_tps,
           "decode_steady_invalid": steady_invalid_reason,
           "n_arrivals": len(arrivals), "wall_s": round(wall, 1), "error": err,
           "output": out_text, "reasoning": reasoning_text,
           "arrivals": arrivals}
with open(OUTFILE, "w") as f:
    json.dump(summary, f, indent=1)
print(f"[{LABEL}] mode={MODE} status={status} pt={pt} ct={ct}\n"
      f"  TTFD={prefill_s}s ({prefill_tps} tok/s prefill)  decode span={decode_tps} "
      f"steady_p50={steady_tps} tok/s  wall={round(wall,1)}s arrivals={len(arrivals)}")
if err:
    print(f"  ERROR: {err[:300]}")
print(f"  output[:300]: {out_text[:300]!r}")
