import importlib.util, sys
spec = importlib.util.spec_from_file_location("r", "day39-stall-reading.py"); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
log = "../spill-a-20260919/pro-single-day30/box/double-park/ev/o1/b05-on/server.log"
L = [ln.rstrip("\n") for ln in open(log) if "[prefix-host] demote" in ln or "contracts door D2H receipt" in ln]
seq2 = [ln for ln in L if "seq=2" in ln or "seq=2," in ln][:4]
print("lines from", log); [print("  ", ln[:160]) for ln in seq2]
def rec(lines):
    run = {"arm": "demote", "run_id": 1, "server_log_lines": lines, "server_demote_ms": [1.0], "intruder": {"fired_at_ms": 1.0, "wall_ms": 1.0}, "tenant_wall_ms": 10.0, "stall_ms": 1.0}
    return {"summary": {"errors": [], "tenant_text_shas": ["x"], "server_demote_ms": [1.0] * 9}, "runs": [run]}
ok_b2, why_b2 = r.admissible("b2", "on", "demote", rec(seq2), "")
nospan = [ln.replace("items=128 (32 KV, 96 f32 spans); ", "") for ln in seq2]
ok_b2n, why_b2n = r.admissible("b2", "on", "demote", rec(nospan), "")
ok_b1n, _ = r.admissible("b1", "on", "demote", rec(nospan), "")
print(f"b2 on day-30 lines: admissible={ok_b2} ({why_b2})")
print(f"b2 on the same lines without the items term: admissible={ok_b2n} ({why_b2n})")
print(f"b1 on the lines without the items term: admissible={ok_b1n}")
good = ok_b2 and not ok_b2n and ok_b1n
print("DAY39 B2-CLAUSE SELFTEST:", "PASS" if good else "FAIL")
sys.exit(0 if good else 1)
