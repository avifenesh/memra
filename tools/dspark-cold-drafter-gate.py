#!/usr/bin/env python3
"""Exactness gate for the DFlash cold-drafter conversion of a vetoed plain hit (memra#1031 second half).

Usage: dspark-cold-drafter-gate.py BASE_URL MODEL OUT.jsonl
Run once against a server with the prefix cache and the DFlash restore doors on (the arm), once
against a server with the prefix cache off (the reference: every request cold-primes DFlash),
then compare the two files with --compare ARM.jsonl REF.jsonl.

Sequence (raw /v1/completions, fixed seed):
  x0: the shared prefix P, served PLAIN (presence_penalty on a greedy request, a shape the
      DFlash route declines) with max_tokens=1, so the cache holds a plain entry for P;
  a:  P + S (S longer than the drafter window), sampled and greedy, short decode: on the arm the
      cache-preferred veto fires on P's plain hit and the conversion re-arms a cold drafter at |P|;
  b:  a's prompt + a's text + a follow-up, which on the arm restores a's tail-carrying entry.
The gate passes when every a/b completion is byte-identical between arm and reference, and
(--check-log on the arm's server log) the arm took the conversion exactly once: the first a step
converts and publishes a tail-carrying boundary entry, every later step restores a tail instead
of being vetoed again (one `cache-preferred` veto, one conversion, at least one `DSPARK restore`
and one `insert (dspark-boundary)`).
GATE_P_PARAS / GATE_S_PARAS size P and S (60-word paragraphs; defaults 10 / 14, about 1.7k / 2.5k
tokens, which fit the 27B on a 24 GB RTX 5090; cold-drafter-gate.sh picks 70 / 80, about 12k / 14k,
on GB10 and on cards that read 32,768 MiB or more, so a 32 GB card, which reads a little under its
nominal size, keeps 10 / 14); S must stay above the drafter window (2,056 rows on the 27B
DFlash2). A refused
request (an HTTP error) prints its status and body and exits 2; --compare prints the prompt sizes
and fails on a missing run file."""
import json
import os
import sys
import urllib.error
import urllib.request


def post(base, body):
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")[:2000]
        print(f"REQUEST REFUSED: HTTP {e.code} on a {len(body['prompt'])}-character prompt: {body_text}",
              file=sys.stderr)
        sys.exit(2)


def para(i, n):
    words = ("memory bandwidth decode kernel scheduler prefix cache draft tail verify accept "
             "token session window boundary restore prime chunk tensor stream").split()
    return " ".join(f"{words[(i * 7 + k * 3) % len(words)]}{(i * 31 + k) % 97}" for k in range(n))


def run(base, model, out):
    P = "Background notes.\n" + "\n".join(para(i, 60) for i in range(int(os.environ.get("GATE_P_PARAS", "10")))) + "\n"
    S = "Session log.\n" + "\n".join(para(1000 + i, 60) for i in range(int(os.environ.get("GATE_S_PARAS", "14")))) + "\n"
    q = "\nQuestion: summarize the notes above in three sentences.\nAnswer:"
    recs = []
    x0 = post(base, {"model": model, "prompt": P, "max_tokens": 1, "temperature": 0,
                     "presence_penalty": 1.5})
    recs.append({"step": "x0", "text": x0["choices"][0]["text"], "usage": x0.get("usage")})
    for mode, temp in (("sampled", 1.0), ("greedy", 0.0)):
        a = post(base, {"model": model, "prompt": P + S + q, "max_tokens": 96,
                        "temperature": temp, "seed": 1234})
        ta = a["choices"][0]["text"]
        recs.append({"step": f"a-{mode}", "text": ta, "usage": a.get("usage")})
        b = post(base, {"model": model, "prompt": P + S + q + ta + "\nNow list two caveats.\n",
                        "max_tokens": 96, "temperature": temp, "seed": 1234})
        recs.append({"step": f"b-{mode}", "text": b["choices"][0]["text"], "usage": b.get("usage")})
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("GATE_RUN_DONE", out)


def compare(arm, ref):
    missing = [f for f in (arm, ref) if not os.path.exists(f)]
    if missing:
        print("missing run file:", " ".join(missing))
        print("GATE FAIL")
        return False
    A = {json.loads(l)["step"]: json.loads(l) for l in open(arm)}
    R = {json.loads(l)["step"]: json.loads(l) for l in open(ref)}
    tokens = {k: (A[k].get("usage") or {}).get("prompt_tokens") for k in ("x0", "a-greedy", "b-greedy") if k in A}
    print("prompt tokens " + " ".join(f"{k}={v}" for k, v in tokens.items()))
    ok = set(A) == set(R) and len(A) == 5
    if not ok:
        print(f"steps differ: arm {sorted(A)} ref {sorted(R)}")
    for k in sorted(A):
        if k == "x0" or k not in R:
            continue
        same = A[k]["text"] == R[k]["text"]
        ok &= same
        print(f"{k}: identical={same} arm_len={len(A[k]['text'])} ref_len={len(R[k]['text'])}")
    print("GATE", "PASS" if ok else "FAIL")
    return ok


def check_log(log):
    """The serving shape behind the a/b sequence, read from the arm's server log."""
    text = open(log, errors="replace").read()
    counts = {
        # Exact success prefixes: "DSPARK cold-drafter restore declined (...)" and the glm5
        # conversion line must not count as a conversion.
        "cache-preferred": text.count("[dspark] cache-preferred:"),
        "cold-drafter restore": text.count("DSPARK cold-drafter restore:"),
        "DSPARK restore": text.count("DSPARK restore:"),
        "insert (dspark-boundary)": text.count("insert (dspark-boundary):"),
    }
    want = {
        "cache-preferred": lambda n: n == 1,
        "cold-drafter restore": lambda n: n == 1,
        "DSPARK restore": lambda n: n >= 1,
        "insert (dspark-boundary)": lambda n: n >= 1,
    }
    ok = True
    for k, n in counts.items():
        good = want[k](n)
        ok &= good
        print(f"shape {k}={n} {'ok' if good else 'WRONG'}")
    print("SHAPE", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
    if sys.argv[1] == "--check-log":
        sys.exit(0 if check_log(sys.argv[2]) else 1)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
