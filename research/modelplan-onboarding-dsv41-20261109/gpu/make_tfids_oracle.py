"""A copy of the oracle whose teacher-forced cases may carry token ids (`tf_ids`) instead of text, so a decode tape's
prompt plus its generated tokens can be scored in one prefill. The program is untouched; the pair (this prefill against
the tape's decode steps) is the vendor's own prefill-vs-decode floor at those positions.

usage: make_tfids_oracle.py ORACLE_DIR OUT_DIR"""
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
old_if = '    if "text" in c:  # teacher-forced'
old_ids = '        ids = tok.encode(c["text"])[: c.get("tf_len", a.tf_len)]'
assert r.count(old_if) == 1 and r.count(old_ids) == 1, "the oracle's teacher-forced branch moved"
r = r.replace(old_if, '    if "text" in c or "tf_ids" in c:  # teacher-forced')
r = r.replace(old_ids, '        ids = list(c["tf_ids"]) if "tf_ids" in c else tok.encode(c["text"])[: c.get("tf_len", a.tf_len)]')
open(p, "w").write(r)
print("wrote", dst)
