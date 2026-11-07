#!/usr/bin/env python3
"""pf_long_memra.py: served long-prompt prefill (TTFT) for memra-server, the memra twin of the
SGLang lane's pf_long.py (same document, needles and question; method copied from its receipt).

  pf_long_memra.py OUT.jsonl --targets 32768 131072 260000 --reps 3 2 1 [--tag ARM]

One user message = a nonce line (unique per boot: target + rep, so no prefix-cache hit, and the
same nonce sequence in every arm so arms see identical prompts) + the synthetic record document
sized with /v1/tokenize + two needles at 10% and 85% depth + the question. Streaming chat
completion, temperature 0, thinking off, presence_penalty 0 (explicit, so the vendor
non-thinking penalty does not reroute the request), max_tokens 40.
ttft_s = first streamed content delta. Rows also carry the text sha256 (the served token
fingerprint for arm identity) and the server-reported prompt token count. stdlib only.
BENCH_BASE (default http://127.0.0.1:8090), BENCH_MODEL (default q38)."""
import argparse, hashlib, json, os, sys, time, urllib.request
from pathlib import Path

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8090").rstrip("/")
MODEL = os.environ.get("BENCH_MODEL", "q38")
HERE = Path(__file__).resolve().parent
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
NEEDLES = (("ALPHA", "TANGERINE-4471"), ("OMEGA", "MARIGOLD-9083"))
QUESTION = ("\n\nTwo lines above give the access codes for vault ALPHA and vault OMEGA. Reply with the two codes, "
            "one per line, ALPHA first, then describe in three sentences how the records are structured.")
KW = {"enable_thinking": False}


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def doc(n_lines, head):
    lines = [f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}."
             for i in range(n_lines)]
    for (vault, code), frac in zip(NEEDLES, (0.10, 0.85)):
        lines.insert(int(n_lines * frac), f"Note: the access code for vault {vault} is {code}.")
    return f"[{head}]\n" + "\n".join(lines) + QUESTION


def count(text):
    r = post("/v1/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": text}],
                              "chat_template_kwargs": KW}, timeout=600)
    for k in ("count", "num_tokens", "length"):
        if k in r:
            return int(r[k])
    return len(r["tokens"])


def lines_for(target, head):
    cache = HERE / "sizes-memra.json"
    sizes = json.loads(cache.read_text()) if cache.exists() else {}
    if str(target) in sizes:
        return sizes[str(target)]
    per = count(doc(2000, head)) / 2000.0
    n = int(target / per)
    for _ in range(6):
        c = count(doc(n, head))
        if abs(c - target) <= 64:
            break
        n += int((target - c) / per)
    sizes[str(target)] = n
    cache.write_text(json.dumps(sizes))
    return n


def stream(messages, max_tokens, timeout=7200):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "presence_penalty": 0.0, "chat_template_kwargs": KW, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None; text = []; finish = None; err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                usage = d.get("usage") or usage
                for ch in d.get("choices", []):
                    finish = ch.get("finish_reason") or finish
                    dl = ch.get("delta", {})
                    piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                    if piece:
                        if first is None:
                            first = time.time()
                        text.append(piece)
    except Exception as e:  # keep the row: a failed request is a result
        err = repr(e)
    t1 = time.time()
    s = "".join(text)
    return {"send_unix": round(t0, 3), "ttft_s": round((first or t1) - t0, 3), "total_s": round(t1 - t0, 3),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"), "finish": finish, "error": err,
            "text_sha256": hashlib.sha256(s.encode()).hexdigest(), "text": s,
            "needles_ok": all(code in s for _, code in NEEDLES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--targets", type=int, nargs="+", default=[32768, 131072, 260000])
    ap.add_argument("--reps", type=int, nargs="+", default=[3, 2, 1])
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-tokens", type=int, default=40)
    a = ap.parse_args()
    reps = a.reps if len(a.reps) == len(a.targets) else [a.reps[0]] * len(a.targets)
    with open(a.out, "a") as f:
        for target, n in zip(a.targets, reps):
            nl = lines_for(target, "fairness-bench-v1")
            for rep in range(n):
                head = f"memra-prefill nonce t{target} r{rep}"
                msgs = [{"role": "user", "content": doc(nl, head)}]
                row = stream(msgs, a.max_tokens)
                row.update({"tag": a.tag, "target": target, "rep": rep, "lines": nl, "nonce": head})
                f.write(json.dumps(row) + "\n"); f.flush()
                print(json.dumps({k: row[k] for k in ("tag", "target", "rep", "ttft_s", "total_s", "prompt_tokens",
                                                      "completion_tokens", "needles_ok", "error", "text_sha256")}),
                      flush=True)


if __name__ == "__main__":
    main()
