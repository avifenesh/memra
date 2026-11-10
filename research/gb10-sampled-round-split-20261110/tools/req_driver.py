#!/usr/bin/env python3
"""The request driver of the sampled-round split (memra#1186).

Six cold requests, one at a time, against a server that is already up: a short greedy warm-up, then
the four 1.2k-token classes of `research/spec-draft-accept-leftovers-20261110/tools/ab_cell.py` (the
same prompts, sampler settings and seeds), then a second greedy 1.2k prompt as the closing control:

  warm      short greedy prompt, 48 tokens (not measured)
  greedy    temperature 0
  t1        temperature 1, fixed seed                 (pure-temperature sampled draft graph)
  chat      temperature 0.7, top_p 0.95, fixed seed   (filtered sampled draft graph)
  pen       temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, fixed seed (eager sampled draft)
  greedy-b  temperature 0, another 1.2k prompt

Per request one JSON line: wall-clock start and end (unix seconds), TTFT, the chunk count, the usage
block (speculative rounds, drafted, accepted), the text sha256 and the text. PAUSE seconds of idle
separate the requests so a profile can tell them apart.

usage: req_driver.py OUT.jsonl PORT [PAUSE_S] [MAX_TOKENS]
"""
import hashlib
import json
import sys
import time
import urllib.request

OUT, PORT = sys.argv[1], int(sys.argv[2])
PAUSE = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
URL = f"http://127.0.0.1:{PORT}"


def notes(topic, n):
    return " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                    f"extends the same record with a new reading." for i in range(1, n + 1))


def long_prompt(topic, n=45):
    return f"{topic.capitalize()} field notes. {notes(topic, n)}\n\nWrite about 300 words on what these notes say about {topic}."


def short_prompt(topic):
    return f"{topic.capitalize()} in brief: write about 300 words on how a {topic} survey is planned and run."


REQS = [
    ("warm", "meadow-short", short_prompt("meadow"), {"temperature": 0}, 48),
    ("greedy", "islands-1k", long_prompt("islands"), {"temperature": 0}, MAX_TOKENS),
    ("t1", "rivers-1k", long_prompt("rivers"), {"temperature": 1.0, "seed": 502}, MAX_TOKENS),
    ("chat", "glaciers-1k", long_prompt("glaciers"), {"temperature": 0.7, "top_p": 0.95, "seed": 512}, MAX_TOKENS),
    ("pen", "lakes-1k", long_prompt("lakes"),
     {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": 532}, MAX_TOKENS),
    ("greedy-b", "deserts-1k", long_prompt("deserts"), {"temperature": 0}, MAX_TOKENS),
]


def stream(content, samp, max_tokens):
    body = {"model": "q", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "stream": True, "chat_template_kwargs": {"enable_thinking": False}, **samp}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"t_start": time.time()}
    t0 = time.perf_counter()
    times, text, usage, err = [], [], None, None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                obj = json.loads(line[5:])
                usage = obj.get("usage") or usage
                for ch in obj.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        times.append(time.perf_counter() - t0)
                        text.append(piece)
    except Exception as exc:  # recorded, never swallowed
        err = repr(exc)
    total = time.perf_counter() - t0
    out.update(t_end=time.time(), total_s=round(total, 4), ttft_s=round(times[0], 4) if times else None,
               chunks=len(times), chunk_times=[round(t, 4) for t in times], usage=usage, error=err,
               text="".join(text))
    return out


def main():
    for cls, pid, content, samp, mt in REQS:
        o = stream(content, samp, mt)
        usage = o.get("usage") or {}
        spec = usage.get("spec") or {}
        toks = usage.get("completion_tokens") or 0
        dec = o["total_s"] - o["ttft_s"] if o["ttft_s"] else None
        row = {"class": cls, "prompt": pid, "sampler": samp, "max_tokens": mt,
               "prompt_tokens": usage.get("prompt_tokens"),
               "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "completion_tokens": toks, "rounds": spec.get("rounds"), "drafted": spec.get("drafted"),
               "accepted": spec.get("accepted"), "acceptance": spec.get("acceptance_rate"),
               "decode_tok_s": round(toks / dec, 2) if dec and toks else None,
               "round_ms": round(1000 * dec / spec["rounds"], 2) if dec and spec.get("rounds") else None,
               "sha": hashlib.sha256(o["text"].encode()).hexdigest()[:16], **o}
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(cls, pid, row["prompt_tokens"], toks, row["rounds"], row["drafted"], row["accepted"],
              row["ttft_s"], row["decode_tok_s"], row["round_ms"], row["sha"], row["error"], flush=True)
        time.sleep(PAUSE)


main()
