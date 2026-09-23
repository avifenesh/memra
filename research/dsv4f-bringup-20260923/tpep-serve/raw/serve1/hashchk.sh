#!/bin/bash
# hashchk.sh: verify every downloaded file against the HF LFS sha256 at the pinned revision.
cd /data/dsv4f/nvfp4
/root/venv/bin/python - <<'PY' > /root/hashchk.log 2>&1
from huggingface_hub import HfApi
import hashlib, os, concurrent.futures as cf
info = HfApi().model_info("tiyuvta/DeepSeek-V4-Flash-0731-NVFP4", revision="bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38", files_metadata=True)
want = {s.rfilename: (s.lfs.sha256 if s.lfs else None, s.size) for s in info.siblings}
def h(n):
    d = hashlib.sha256()
    with open(n, "rb") as f:
        while b := f.read(1 << 24): d.update(b)
    return n, d.hexdigest()
bad = 0
lfs = [n for n, (s, _) in want.items() if s]
with cf.ThreadPoolExecutor(16) as ex:
    for n, got in ex.map(h, lfs):
        ok = got == want[n][0]; bad += not ok
        print(("OK " if ok else "BAD ") + n + " " + got)
for n, (s, size) in want.items():
    if not s:
        ok = os.path.getsize(n) == size; bad += not ok
        print(("OK-size " if ok else "BAD-size ") + n)
print(f"HASHCHK files={len(want)} lfs={len(lfs)} bad={bad}")
PY
echo HASH_DONE >> /root/hashchk.log
