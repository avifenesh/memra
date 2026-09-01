#!/usr/bin/env python3
"""Materialise the run-gen prompt files from the real chat payloads. Real prompts only."""
import json
for n in ("curve-0400", "curve-1000"):
    d = json.load(open("/root/%s.json" % n))
    t = d[0] if isinstance(d, list) else d["messages"][0]["content"]
    open("/root/s37h-%s.prompt" % n, "w").write(t)
    print("  wrote /root/s37h-%s.prompt (%d chars)" % (n, len(t)))
