#!/usr/bin/env python3
"""Verify the published repo against the banked manifest. Token from HF_TOKEN."""
import os, sys, pathlib
from huggingface_hub import HfApi

REPO = "Avifenesh/GLM-5.3-Flash-NVFP4"
MANIFEST = pathlib.Path.home() / "glm53-hf-lane" / "SHA256SUMS.txt"
NEVER = {"README.md", "config.json.pre-keeplist-fix"}

want = {}
for line in MANIFEST.read_text().splitlines():
    if not line.strip():
        continue
    h, name = line.split(None, 1)
    name = name.strip()
    if name not in NEVER:
        want[name] = h

api = HfApi(token=os.environ["HF_TOKEN"])
info = api.model_info(REPO, files_metadata=True)
got = {s.rfilename: s for s in info.siblings}

bad = []
for name, h in sorted(want.items()):
    s = got.get(name)
    if s is None:
        bad.append(f"MISSING {name}")
        continue
    remote = (s.lfs.get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)) if s.lfs else None
    if remote is None:
        print(f"  {name}: present, size {s.size} (no lfs sha exposed)")
    elif remote != h:
        bad.append(f"HASH MISMATCH {name}: local {h} remote {remote}")
    else:
        print(f"  {name}: sha256 MATCH")

extra = sorted(set(got) - set(want) - {"README.md", ".gitattributes"})
print("expected files:", len(want), "| present:", len(got))
if extra:
    print("EXTRA FILES IN REPO:", extra)
    bad.append(f"extra files: {extra}")
for b in bad:
    print("FAIL:", b)
sys.exit(1 if bad else 0)
