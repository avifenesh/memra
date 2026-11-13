"""A copy of an oracle directory whose generation cases may take the token of every step ("force_ids") instead of the
argmax, and whose tapes also keep every step's log-sum-exp and argmax.

The vendor's own decode program, teacher-forced: step 0 scores the prompt's last position, step k > 0 is the decode call
that feeds force_ids[k - 1] (the vendor `generate.py` loop with the token chosen from the list). Fed the tokens of a
greedy tape, the copy of the tape's own oracle rewrites that tape (same kernels, same calls), now with lse; the copy of
another valid realization (the global-max attention oracle, make_globalmax_oracle.py) gives the vendor's decode floor
at the same steps.

usage: make_forced_oracle.py ORACLE_DIR OUT_DIR"""
import os
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.endswith(".py") or f.endswith(".json") or f.endswith(".i32"):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
p = os.path.join(dst, "run_oracle.py")
r = open(p).read()
edits = [
    ("    gen, tops, full = [], [], []\n", "    gen, tops, full, lses, amax = [], [], [], [], []\n"),
    ("        for step in range(a.max_new):\n",
     '        for step in range(len(c["force_ids"]) if "force_ids" in c else a.max_new):\n'),
    ("            tops.append((v.clone(), i.clone()))\n",
     "            tops.append((v.clone(), i.clone()))\n"
     "            lses.append(float(torch.logsumexp(lg.double(), 0)))\n"
     "            amax.append(int(lg.argmax()))\n"),
    ("            t_id = int(lg.argmax())\n",
     '            t_id = int(c["force_ids"][step]) if "force_ids" in c else int(lg.argmax())\n'),
    ('"prefill_s": t_pre, "total_s": dt}',
     '"prefill_s": t_pre, "total_s": dt,\n           "lse": torch.tensor(lses, dtype=torch.float64), "argmax": torch.tensor(amax)}'),
]
for old, new in edits:
    assert r.count(old) == 1, f"the oracle's generation loop moved: {old!r}"
    r = r.replace(old, new)
open(p, "w").write(r)
print("wrote", dst)
