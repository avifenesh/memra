#!/usr/bin/env python3
"""Greedy + sampled coherence probe against a serving arm (OpenAI-compatible chat API).
Usage: coherence.py BASE_URL MODEL OUT.jsonl
Eight fixed prompts, each at T=0 (greedy, 384 tokens) and at T=1 seed 1234 (384 tokens);
writes one record per (prompt, mode) with the text and completion token count."""
import json
import sys
import urllib.request

BASE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a Rust function that returns the n-th Fibonacci number iteratively, then explain its complexity.",
    "A train leaves at 14:05 and arrives at 17:50. How long is the trip in minutes? Show the steps.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Write a bash one-liner that counts the lines of every .py file under the current directory, sorted.",
    "What is the difference between a process and a thread? Answer in three short paragraphs.",
    "Translate to Hebrew: 'The meeting moved to Thursday at ten in the morning.'",
    "Write a SQL query that returns the top 3 customers by total order value from tables customers(id, name) and orders(id, customer_id, amount).",
]


def chat(prompt, temperature):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 384,
        "temperature": temperature,
        "seed": 1234,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"]


with open(OUT, "w") as f:
    for i, p in enumerate(PROMPTS):
        for mode, t in (("greedy", 0.0), ("sampled", 1.0)):
            text, n = chat(p, t)
            f.write(json.dumps({"prompt": i, "mode": mode, "tokens": n, "text": text}) + "\n")
            f.flush()
print("COHERENCE_DONE", OUT)
