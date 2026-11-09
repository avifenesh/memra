"""Add the tilelang-schedule attention reference (`o_sched`) to existing sattn_* fixture dirs (same function as
make_fixtures.py; avoids regenerating every case)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import torch
torch.set_num_threads(8)
src = open(os.path.join(HERE, "make_fixtures.py")).read()
a = src.index("def sattn_schedule(")
b = src.index("# ---------------------------------------------------------------- sink sparse attention")
ns = {"torch": torch}
exec(src[a:b], ns)
from fixio import Fix
DT = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32}
root = sys.argv[1]
for case in sorted(os.listdir(root)):
    if not case.startswith("sattn_"):
        continue
    d = os.path.join(root, case)
    idx = json.load(open(os.path.join(d, "index.json")))
    t = {k: torch.from_file(os.path.join(d, e["file"]), size=os.path.getsize(os.path.join(d, e["file"])), dtype=torch.uint8).view(DT[e["dtype"]]).reshape(e["shape"]) for k, e in idx["tensors"].items()}
    fx = Fix(d)
    fx.idx = idx
    fx.put("o_sched", ns["sattn_schedule"](t["q"], t["kv"], t["sink"], t["idx"], idx["meta"]["scale"]))
    fx.close()
    print(case, "o_sched added", flush=True)
