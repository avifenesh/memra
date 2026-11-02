#!/usr/bin/env python3
"""Freeze a long multi-turn chat (~131k tokens on the Qwen3.5 tokenizer) from repo docs.
usage: make_longchat.py TOTAL_CHARS OUT.json"""
import json, sys
from pathlib import Path
wt = Path.home() / ".cache/memra-relprofile/wt"
total, out = int(sys.argv[1]), sys.argv[2]
text = "".join((wt / p).read_text() for p in
               ["docs/FLAGS.md", "docs/PERFORMANCE.md", "ARCHITECTURE.md", "docs/TESTING.md", "docs/SERVING.md"])
text = text[:total]
turn = 12000
msgs = [{"role": "system", "content": "You are a careful engineering assistant."}]
for i in range(0, len(text), turn):
    msgs.append({"role": "user" if (i // turn) % 2 == 0 else "assistant", "content": text[i:i + turn]})
if msgs[-1]["role"] != "user":
    msgs.append({"role": "user", "content": "Summarize the discussion above in one line."})
Path(out).write_text(json.dumps(msgs, ensure_ascii=False))
print(len(msgs), "messages", len(text), "chars")
