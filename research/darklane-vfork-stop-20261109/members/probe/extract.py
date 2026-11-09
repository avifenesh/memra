#!/usr/bin/env python3
"""Copy the acknowledgement code of darklane.rs verbatim into the probe crate.

usage: extract.py DARKLANE_RS OUT_RS
From the doc comment of child_stop_acknowledged to the end of leader_stop_took_effect.
"""
import hashlib
import sys

src = open(sys.argv[1]).read()
start = src.index("/// Observe a stop without reaping an exit")
end_fn = src.index("fn leader_stop_took_effect(")
end = src.index("\n}\n", end_fn) + 3
body = src[start:end]
open(sys.argv[2], "w").write(body)
print("darklane.rs", hashlib.sha256(src.encode()).hexdigest())
print("extracted", hashlib.sha256(body.encode()).hexdigest(), len(body.splitlines()), "lines")
