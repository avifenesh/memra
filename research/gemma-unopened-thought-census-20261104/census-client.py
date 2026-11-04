#!/usr/bin/env python3
"""E4B unopened-thought census for the gemma4 parser rule (GEMMA-UNOPENED-THOUGHT).
usage: gemma-census-client.py PORT OUTDIR
For each prompt, thinking off and thinking on (reasoning_effort low and medium), greedy, non-stream and stream:
records reasoning/content bytes, tag leaks in stream deltas, and stream == non-stream reassembly."""
import json
import pathlib
import sys
import urllib.request

PROMPTS = [
    "Reply with exactly: ok",
    "What is 17 times 23? Show your work briefly.",
    "Explain in about 150 words how the water cycle works.",
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip?",
    "Write a haiku about a lighthouse.",
    "List three differences between TCP and UDP.",
    "Is 391 a prime number? Explain.",
    "Translate to French: The meeting moved to Thursday afternoon.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What causes the seasons on Earth?",
    "Sort these words alphabetically: pear, apple, mango, kiwi, banana.",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Give me a regular expression that matches an ISO date like 2026-11-04.",
    "Which is heavier, a kilogram of feathers or a kilogram of iron? Why?",
    "Plan a three-step morning routine for someone who works night shifts.",
    "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
    "Write a SQL query that counts orders per customer from a table orders(customer_id, id).",
    "What is the capital of Australia, and why is it not Sydney?",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "Name four planets in order from the Sun.",
    "Debug this: for i in range(10) print(i)",
    "Explain what a hash map is to a ten-year-old.",
    "How many seconds are in a leap year?",
]
TAGS = ("<turn|>", "<|channel>", "<channel|>", "<|turn>")


def post(port, body, stream):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(dict(body, stream=stream)).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read().decode()


def main():
    port, out = int(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(PROMPTS):
        for mode in ("off", "low", "medium"):
            body = {"model": "e4b", "messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0}
            if mode != "off":
                body["reasoning_effort"] = mode
            ns = post(port, body, False)
            st = post(port, body, True)
            (out / f"p{i:02d}-{mode}-ns.json").write_text(ns)
            (out / f"p{i:02d}-{mode}-st.sse").write_text(st)
            m = json.loads(ns)["choices"][0]["message"]
            want_c, want_r = m.get("content") or "", m.get("reasoning") or ""
            c = r = ""
            leak = []
            for line in st.splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])["choices"][0]["delta"]
                dc = d.get("content") or ""
                leak += [t for t in TAGS if t in dc]
                c += dc
                r += d.get("reasoning") or ""
            ns_leak = [t for t in TAGS if t in want_c]
            row = dict(p=i, mode=mode, reasoning_bytes=len(want_r.encode()), content_bytes=len(want_c.encode()),
                       stream_equal=(c == want_c and r == want_r), stream_tag_leak=sorted(set(leak)),
                       nonstream_tag_leak=ns_leak)
            rows.append(row)
            print(json.dumps(row), flush=True)
    (out / "census.json").write_text(json.dumps(rows, indent=1))
    think = [x for x in rows if x["reasoning_bytes"] > 0]
    print("SUMMARY", json.dumps(dict(
        cells=len(rows), with_reasoning=len(think),
        max_reasoning_bytes=max((x["reasoning_bytes"] for x in rows), default=0),
        over_256=sum(x["reasoning_bytes"] >= 256 for x in rows),
        stream_mismatch=[(x["p"], x["mode"]) for x in rows if not x["stream_equal"]],
        stream_leaks=[(x["p"], x["mode"], x["stream_tag_leak"]) for x in rows if x["stream_tag_leak"]],
        nonstream_leaks=[(x["p"], x["mode"]) for x in rows if x["nonstream_tag_leak"]],
        low_effort_thinking_prompts=[x["p"] for x in rows if x["mode"] == "low" and x["reasoning_bytes"] > 0])))


main()
