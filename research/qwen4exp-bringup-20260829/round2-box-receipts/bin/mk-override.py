"""Mint a rope-override twin of an artifact dir: every file HARDLINKED, config.json rewritten.

HARDLINK, not symlink: the loader's snapshot containment check REFUSES symlinks (YARN-CELL
§1 — "Symlinks are refused by the loader's snapshot containment check; hardlinks are the
working form"). This script shipped with `os.symlink` anyway, so as committed it produced a
dir the engine would not load; round 2 hit it on the replacement box. Hardlinks cost no
disk (same inodes as the 174 GB mint) and pass the check.
"""

import json, os, sys

src, dst, factor, mpe = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
os.makedirs(dst, exist_ok=True)
linked = 0
for f in sorted(os.listdir(src)):
    s, p = os.path.join(src, f), os.path.join(dst, f)
    # Skip config.json (rewritten below), anything already present, and DIRECTORIES:
    # a downloaded mint carries an hf `.cache/` dir, and os.link refuses directories
    # (EPERM) — which aborted the whole mint on the round-2 box. The engine reads only
    # the top-level files, so dropping the cache dir is correct, not a workaround.
    if f == "config.json" or os.path.lexists(p) or not os.path.isfile(s):
        continue
    os.link(s, p)
    linked += 1
c = json.load(open(os.path.join(src, "config.json")))
tc = c["text_config"]
rp = tc.setdefault("rope_parameters", {})
rp["rope_type"] = "yarn"
rp["factor"] = factor
rp["original_max_position_embeddings"] = 262144
tc["max_position_embeddings"] = mpe
json.dump(c, open(os.path.join(dst, "config.json"), "w"), indent=1)
print(f"wrote {dst}: {linked} files hardlinked, factor {factor}, mpe {mpe}")
