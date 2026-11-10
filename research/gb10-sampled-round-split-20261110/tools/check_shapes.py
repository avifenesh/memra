#!/usr/bin/env python3
"""check_shapes.py AB_CELL REQ_DRIVER: the four 1.2k requests of req_driver.py carry the same prompt
text and sampler settings as the 1k rows of the cell program (ab_cell.py PERF). Both files are read
as source up to their first network or process code, so nothing is sent or started."""
import sys

ab_src = open(sys.argv[1]).read().split("\ndef boot(")[0]
rq_src = open(sys.argv[2]).read().split("\ndef stream(")[0]
ab, rq = {}, {}
sys.argv = ["x", "out", "bin", "arm", "18196"]
exec(ab_src.replace('OUT.mkdir(parents=True, exist_ok=True)', '').replace('(OUT / "texts").mkdir(exist_ok=True)', ''), ab)
sys.argv = ["x", "out.jsonl", "18196"]
exec(rq_src, rq)
cell = {cls: (content, samp) for cls, pid, content, samp in ab["PERF"]}
ok = True
for cls, pid, content, samp, mt in rq["REQS"]:
    key = {"greedy": "greedy-1k", "t1": "t1-1k", "chat": "chat-1k", "pen": "pen-1k"}.get(cls)
    if key is None:
        print(f"{cls:9s} {pid:13s} not a cell row (warm-up or closing control), {len(content)} chars, {samp}")
        continue
    same = cell[key] == (content, samp)
    ok &= same
    print(f"{cls:9s} {pid:13s} against {key:10s}: prompt and sampler {'equal' if same else 'DIFFER'}, {len(content)} chars, {samp}, max_tokens {mt}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
