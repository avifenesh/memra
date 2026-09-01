#!/usr/bin/env python3
"""Materialise the run-spec prompt file from the real chat payload. Real prompts only."""
import json

d = json.load(open("/root/curve-0400.json"))
t = d[0] if isinstance(d, list) else d["messages"][0]["content"]
open("/root/dcw-0400.prompt", "w").write(t)
print("wrote /root/dcw-0400.prompt (%d chars)" % len(t))
