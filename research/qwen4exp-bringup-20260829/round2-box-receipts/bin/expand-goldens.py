#!/usr/bin/env python3
"""Expand the in-repo `hidden-goldens.pt` into the raw f32 bins `qwen4exp_real_gate
--goldens` reads — WITHOUT the 336 GB BF16 checkpoint.

Why this exists: `prep-real-gate.py` builds the same dump but imports transformers and
loads the BF16 tokenizer, purely to re-tokenize `greedy-goldens.json` into `prompts.tsv`.
That made a lost box mean a 336 GB re-download (it did, twice, on 2026-08-29..31). But
`prompts.tsv`, `input-ids.txt` and `manifest.tsv` are all mirrored in this lane next to
`hidden-goldens.pt`, so the ONLY thing a replacement box actually has to recompute is the
tensor->bin expansion, which needs torch and nothing else.

This script therefore:
  1. expands every record in hidden-goldens.pt to `<name>.bin` (little-endian f32,
     row-major, batch dim squeezed) — byte-for-byte the format prep-real-gate.py writes;
  2. RE-DERIVES manifest.tsv and input-ids.txt and hard-compares them against the
     mirrored copies already in the dump dir, so a silently different goldens .pt fails
     here instead of surfacing as a bogus gate delta (loud-failure law).

Usage: expand-goldens.py <dump_dir>
  <dump_dir> already holds hidden-goldens.pt + the mirrored manifest.tsv / input-ids.txt
  / prompts.tsv; the .bin files land beside them.
"""

import os
import sys

import torch

if len(sys.argv) != 2:
    sys.exit(__doc__)
dump = sys.argv[1]

cap = torch.load(
    os.path.join(dump, "hidden-goldens.pt"), map_location="cpu", weights_only=True
)
ids = cap.pop("input_ids").flatten().tolist()

rows = []
for name, t in sorted(cap.items()):
    t = t.float()
    if t.dim() == 3:
        assert t.shape[0] == 1, (name, t.shape)
        t = t.squeeze(0)
    assert t.dim() == 2, (name, t.shape)
    fn = f"{name}.bin"
    t.contiguous().numpy().astype("<f4").tofile(os.path.join(dump, fn))
    rows.append(f"{name}\t{t.shape[0]}\t{t.shape[1]}\t{fn}")

derived_manifest = "\n".join(rows) + "\n"
derived_ids = " ".join(str(i) for i in ids)

# --- the two hard compares against the mirrored copies (never "trust and overwrite") ---
failures = []
mpath = os.path.join(dump, "manifest.tsv")
if os.path.exists(mpath):
    with open(mpath) as f:
        banked = f.read()
    if banked != derived_manifest:
        failures.append(
            "manifest.tsv MISMATCH vs the records in hidden-goldens.pt "
            f"(banked {len(banked.splitlines())} lines, derived {len(rows)})"
        )
else:
    with open(mpath, "w") as f:
        f.write(derived_manifest)
    print(f"manifest.tsv: WROTE (no mirrored copy present), {len(rows)} records")

ipath = os.path.join(dump, "input-ids.txt")
if os.path.exists(ipath):
    with open(ipath) as f:
        banked_ids = f.read().split()
    if banked_ids != [str(i) for i in ids]:
        failures.append(
            f"input-ids.txt MISMATCH: banked {len(banked_ids)} ids, "
            f"hidden-goldens.pt probe has {len(ids)}"
        )
else:
    with open(ipath, "w") as f:
        f.write(derived_ids)
    print(f"input-ids.txt: WROTE (no mirrored copy present), T={len(ids)}")

if not os.path.exists(os.path.join(dump, "prompts.tsv")):
    failures.append(
        "prompts.tsv is MISSING and this script cannot mint it (it needs the BF16 "
        "tokenizer via prep-real-gate.py) — copy the mirrored lane file instead"
    )

if failures:
    for line in failures:
        print(f"FAIL: {line}", file=sys.stderr)
    sys.exit(1)

total = sum(os.path.getsize(os.path.join(dump, r.split("\t")[3])) for r in rows)
print(
    f"expanded {len(rows)} records ({total / 2**20:.1f} MiB of f32 bins), probe T={len(ids)}; "
    "manifest.tsv + input-ids.txt VERIFIED identical to the mirrored copies"
)
