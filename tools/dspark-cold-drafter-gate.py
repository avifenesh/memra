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
GATE_P_PARAS / GATE_S_PARAS size P and S (60-word paragraphs; defaults 70 / 80, about 12k / 14k
tokens); S must stay above the drafter window (2,056 rows on the 27B DFlash2)."""
import json
import os
import sys
import urllib.request


def post(base, body):
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def para(i, n):
    words = ("memory bandwidth decode kernel scheduler prefix cache draft tail verify accept "
             "token session window boundary restore prime chunk tensor stream").split()
    return " ".join(f"{words[(i * 7 + k * 3) % len(words)]}{(i * 31 + k) % 97}" for k in range(n))


def run(base, model, out):
    P = "Background notes.\n" + "\n".join(para(i, 60) for i in range(int(os.environ.get("GATE_P_PARAS", "70")))) + "\n"
    S = "Session log.\n" + "\n".join(para(1000 + i, 60) for i in range(int(os.environ.get("GATE_S_PARAS", "80")))) + "\n"
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
    A = {json.loads(l)["step"]: json.loads(l) for l in open(arm)}
    R = {json.loads(l)["step"]: json.loads(l) for l in open(ref)}
    ok = True
    for k in sorted(A):
        if k == "x0":
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
        "cache-preferred": text.count("cache-preferred"),
        "cold-drafter restore": text.count("cold-drafter restore"),
        "DSPARK restore": text.count("DSPARK restore"),
        "insert (dspark-boundary)": text.count("insert (dspark-boundary)"),
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
