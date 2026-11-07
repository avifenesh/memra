#!/usr/bin/python3
"""Re-grade a quality-panel records.jsonl offline with the panel's own judge.

The panel run under the agent harness's venv python graded every HumanEval program with
`sys.executable` inside the bwrap sandbox, where that venv path does not exist, so all 164
programs failed with `bwrap: execvp ... No such file or directory` before running. Generation
is unaffected (the texts are in the records). Run this with the system python so the sandbox
can execute it:  /usr/bin/python3 regrade_panel.py RECORDS OUT
GSM8K rows are re-scored the same way (string answer extraction, no sandbox)."""
import json
import sys

sys.path.insert(0, "~/.cache/spark-cast-qualification-20261024-2250")
sys.path.insert(0, "~/.cache/model-scout/harness")
import client  # noqa: E402
from quality_protocol import load_jobs  # noqa: E402

rec_path, out_path = sys.argv[1], sys.argv[2]
jobs = {(j["task"], j["id"]): j for j in load_jobs()}
rows = [json.loads(line) for line in open(rec_path) if line.strip()]
out = []
tally = {}
for r in rows:
    job = jobs[(r["task"], r["id"])]
    item = job["item"]
    if r["task"] == "long_humaneval":
        ok, err = client.judge.run_he_program(item["prompt"], client.judge.extract_code(r["text"] or ""),
                                              item["test"], item["entry_point"])
        g = {"pass": ok, "error": err}
    else:
        gold = float(item["answer"].split("####")[-1].strip().replace(",", ""))
        got = client.judge.gsm_final(r["text"] or "")
        g = {"gold": gold, "got": got, "pass": got is not None and abs(gold - got) < 1e-6}
    row = dict(r, **g, regraded_by=sys.executable)
    out.append(row)
    t = tally.setdefault(r["task"], [0, 0])
    t[0] += bool(g["pass"])
    t[1] += 1
with open(out_path, "w") as f:
    for row in out:
        f.write(json.dumps(row) + "\n")
print(json.dumps({task: f"{p}/{n}" for task, (p, n) in tally.items()}))
