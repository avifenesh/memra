#!/usr/bin/env python3
"""eos_ab.py OUT.jsonl [--long] [--tag ARM]: one served cell for the EOS burst-end A/B (round 5 lever 1).

Requests, in order, on a fresh server (the pf_long_memra.py document, needles and question):
  g1k      greedy, ~1k-token prompt, single turn
  g1k-t2   greedy second turn of g1k: the first answer as the assistant turn, then a short user
           message (the parked spec session's resume path)
  s1k      sampled (temperature 0.8, top_p 0.95, seed 1234), its own ~1k nonce
  g131k    greedy ~131k-token prompt (with --long)
Each row: TTFT, total, completion tokens, finish and the text sha256 (the arms' identity check).
512 max tokens: every answer here ends on EOS. BENCH_BASE / BENCH_MODEL as pf_long_memra.py."""
import argparse, hashlib, json, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_long_memra as pf  # noqa: E402


def stream(messages, max_tokens, sampling):
    body = {"model": pf.MODEL, "messages": messages, "max_tokens": max_tokens, "presence_penalty": 0.0,
            "chat_template_kwargs": pf.KW, "stream": True, "stream_options": {"include_usage": True}, **sampling}
    req = urllib.request.Request(pf.BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None; text = []; finish = None; err = None
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
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
            "text_sha256": hashlib.sha256(s.encode()).hexdigest(), "text": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    greedy = {"temperature": 0.0}
    sampled = {"temperature": 0.8, "top_p": 0.95, "seed": 1234}
    nl = pf.lines_for(1024, "fairness-bench-v1")
    with open(a.out, "a") as f:
        def put(name, row):
            row.update({"tag": a.tag, "req": name})
            f.write(json.dumps(row) + "\n"); f.flush()
            return row
        m1 = [{"role": "user", "content": pf.doc(nl, "memra-eos nonce g1k")}]
        r1 = put("g1k", stream(m1, 512, greedy))
        m2 = m1 + [{"role": "assistant", "content": r1["text"]},
                   {"role": "user", "content": "Now give the two codes again in reverse order, OMEGA first, and nothing else."}]
        put("g1k-t2", stream(m2, 512, greedy))
        put("s1k", stream([{"role": "user", "content": pf.doc(nl, "memra-eos nonce s1k")}], 512, sampled))
        if a.long:
            nll = pf.lines_for(131072, "fairness-bench-v1")
            put("g131k", stream([{"role": "user", "content": pf.doc(nll, "memra-eos nonce g131k")}], 512, greedy))


if __name__ == "__main__":
    main()
