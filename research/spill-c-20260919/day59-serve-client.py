#!/usr/bin/env python3
"""Day 59 G3 client (DAY59.md section 1): three sequential greedy completions, then four concurrent ones, 48 tokens
each, against one boot; every response saved whole. usage: day59-serve-client.py <port> <out-dir>"""
import json
import sys
import threading
import urllib.request
from pathlib import Path

PROMPTS = [
    "Explain, step by step, how a tide gauge converts water pressure into a height reading, and name two error sources.",
    "Write a short checklist for commissioning a small hydroelectric turbine hall, one line per item.",
    "Summarize the differences between a cache that admits on first miss and one that admits on second miss.",
    "List four ways a scheduler can starve a long request, and one fix for each.",
    "Describe what a copy engine does on a GPU and why overlapping it with compute can save time.",
    "Give three reasons a benchmark run on a laptop can read slower than the same run on a workstation.",
    "Explain what a checksum protects against and what it does not, in five sentences.",
]


def complete(port, prompt, out):
    body = {"model": "gate", "prompt": prompt, "max_tokens": 48, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out.write_text(r.read().decode())
    except Exception as err:  # recorded, never swallowed: the reader counts it
        out.write_text(json.dumps({"error": repr(err)}))


def main():
    port, d = sys.argv[1], Path(sys.argv[2])
    d.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(PROMPTS[:3]):
        complete(port, p, d / f"seq{i}.json")
    threads = [threading.Thread(target=complete, args=(port, p, d / f"conc{i}.json")) for i, p in enumerate(PROMPTS[3:])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
