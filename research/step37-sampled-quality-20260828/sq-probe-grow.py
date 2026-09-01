# Grow-panic discriminator: one big fresh prime (through user turn 7) on a session,
# then ONE grow (full conversation through user turn 8) on the same session.
# If this panics: the grow bug is target-size-triggered (>6.2k rows unreachable).
# If it survives: the bug is cumulative-grow-triggered, and this 2-request shape is a
# viable degraded-mode warm instrument for turn 8.
import json, os, sys

sys.path.insert(0, "/root/sq")
import importlib.util
spec = importlib.util.spec_from_file_location("sqd", "/root/sq/sq-drive.py")
sqd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sqd)

tr = json.load(open("/root/sq/transcript.json"))
U = {int(k): v for k, v in tr["U"].items()}
A = {int(k): v for k, v in tr["A"].items()}

sid = "sq-probeA"
r1, e1 = sqd.guarded(sqd.msgs_through(U, A, 7), sid, 64)
print("REQ1 (through user7, fresh):", json.dumps({k: r1[k] for k in
      ("ttft", "finish", "err", "eng_fresh", "eng_suffix", "walk_fresh", "walk_suffix",
       "suffix_tokens", "eng_lines", "walk_lines")} if r1 else {"err": e1}), flush=True)
r2, e2 = sqd.guarded(sqd.msgs_through(U, A, 8), sid, 64)
print("REQ2 (through user8, one grow):", json.dumps({k: r2[k] for k in
      ("ttft", "finish", "err", "eng_fresh", "eng_suffix", "walk_fresh", "walk_suffix",
       "suffix_tokens", "eng_lines", "walk_lines")} if r2 else {"err": e2}), flush=True)
ok2 = r2 and not r2["err"] and (r2["reasoning"] + r2["content"]).strip()
print("PROBE_VERDICT=%s" % ("ONE_GROW_SURVIVES" if ok2 else "ONE_GROW_DIES"), flush=True)
