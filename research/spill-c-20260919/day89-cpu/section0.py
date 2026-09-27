#!/usr/bin/python3
"""DAY89/DAY90 section 0: the governor's and the retire side's in-situ figures at I21 and I22, read from the committed
split logs (rtx5090-day85/split22/ev) through day83-read.py's own terms; ack_release is computed there but not printed."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("day83_read", HERE / "day83-read.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
ev = HERE / "rtx5090-day85/split22/ev"
KEYS = ("bank_stage_charge_ns", "bank_ack_release_ns", "bank_ack_ns", "bank_retire_ns", "bank_collect_ns",
        "retire_outer_ns", "eng_retire_ns", "eng_wait_ns", "pf_retire_ns")
for arm in ("i21s", "i22s"):
    for phase, a, b in (("generate", "gate", "generate"), ("window", "warm", "window")):
        runs = [r.parse(p) for p in sorted(ev.glob(f"o[12]-{arm}-r*.log"))]
        per = [r.terms(x, a, b) for x in runs]
        vals = " ".join(f"{k[:-3]}={r.us_tok(r.med([t[k] for t in per]))}" for k in KEYS)
        print(f"DAY89 S0 rig=rtx5090 {arm} {phase} N={len(per)} (us per token, medians): {vals}")
