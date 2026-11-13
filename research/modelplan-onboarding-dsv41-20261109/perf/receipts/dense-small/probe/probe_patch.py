#!/usr/bin/env python3
"""Probe-only: env overrides of the dense small-row body's launch shape in the staged copy of the new dsv41_gpu.cu
(V41D_RPB: rows a block at least, V41D_KB: in-flight KB a SM). Never in the engine source."""
import sys
p = sys.argv[1]
s = open(p).read()
old = "    const long per_sm = (n + sms - 1) / sms;\n"
new = ("    long per_sm = (n + sms - 1) / sms;\n"
       "    { const char* e = getenv(\"V41D_RPB\"); if (e && atol(e) > per_sm) per_sm = atol(e); }\n")
assert old in s
s = s.replace(old, new)
old = "    const long want = 48L * 1024 / bps;\n"
new = ("    long want = 48L * 1024 / bps;\n"
       "    { const char* e = getenv(\"V41D_KB\"); if (e && atol(e) > 0) want = atol(e) * 1024L / bps; }\n")
assert old in s
s = s.replace(old, new)
if "#include <stdlib.h>" not in s:
    s = s.replace("#include <stdint.h>\n", "#include <stdint.h>\n#include <stdlib.h>\n", 1)
open(p, "w").write(s)
print("patched")
