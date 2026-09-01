#!/usr/bin/env python3
"""Mint the long-context ladder corpus (`--ladder-ids`) from the memra tree's own source.

The yarn cell's ladder ran on `ids=1150000` pre-tokenized tokens of REAL text (its
continuations were graded as "valid Rust/CUDA source continuations of the real corpus").
That ids file only ever existed on the box, so both spot reclaims (2026-08-29, -08-31)
took it with them and round 2 had to re-derive it. It is now reproducible: the corpus is
the memra checkout, walked in a PINNED deterministic order, and the only external input
is the artifact's own `tokenizer.json`.

Determinism (this matters — the ladder's continuation-coherence rows are only comparable
across rounds if the fed tokens are the same tokens):
  * file set = the globs below, resolved against <memra_root>, sorted by POSIX path;
  * each file contributes a `// ==== <relpath> ====` banner then its verbatim text;
  * concatenation order is that sorted order, and it does not depend on the filesystem;
  * tokenization is one `tokenizer.encode` over the joined text (no chunk boundaries),
    then truncated to <n_tokens>.
Because the file set is the working tree, the receipt line printed at the end names the
commit the corpus was minted at — quote it beside any ladder receipt.

Usage: make-ladder-ids.py <memra_root> <tokenizer_dir> <out.txt> [n_tokens]
  <tokenizer_dir>  a dir holding tokenizer.json (the NVFP4 artifact has one)
  n_tokens         default 1_150_000 (the yarn cell's width: 1M rung + in-context
                   continuations + the ladder's own headroom)
"""

import glob
import os
import subprocess
import sys

from tokenizers import Tokenizer

# Rust + CUDA + the engine's own docs: the mix the yarn cell's continuations were graded
# against. Ordered here for readability only; the effective order is the sorted union.
GLOBS = [
    "crates/*/src/**/*.rs",
    "crates/*/tests/**/*.rs",
    "crates/*/src/**/*.cu",
    "crates/*/src/**/*.cuh",
    "kernels/**/*.cu",
    "kernels/**/*.cuh",
    "docs/**/*.md",
]

if len(sys.argv) not in (4, 5):
    sys.exit(__doc__)
root, tokdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
want = int(sys.argv[4]) if len(sys.argv) == 5 else 1_150_000

paths = set()
for g in GLOBS:
    for p in glob.glob(os.path.join(root, g), recursive=True):
        if os.path.isfile(p):
            paths.add(os.path.relpath(p, root))
if not paths:
    sys.exit(f"FAIL: no corpus files matched under {root}")
paths = sorted(paths)

parts = []
for rel in paths:
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        continue
    parts.append(f"// ==== {rel} ====\n{text}")
joined = "\n".join(parts)

tok = Tokenizer.from_file(os.path.join(tokdir, "tokenizer.json"))
ids = tok.encode(joined, add_special_tokens=False).ids
if len(ids) < want:
    sys.exit(
        f"FAIL: corpus tokenized to {len(ids)} ids, need {want} — widen GLOBS "
        "(do NOT repeat the corpus: a repeated prefix makes the depth rungs measure "
        "cache-friendly self-similar text instead of real long context)"
    )
ids = ids[:want]

with open(out, "w") as f:
    # One id per line is 8 MB of file at 1.15M ids; comma-joined is half that and the
    # gate's parser splits on whitespace OR comma.
    f.write(",".join(map(str, ids)))
    f.write("\n")

try:
    commit = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
except Exception:
    commit = "unknown"
print(
    f"ladder-ids: {len(ids)} tokens from {len(parts)} files "
    f"({len(joined)} chars) -> {out}\n"
    f"corpus_commit={commit}  first8={ids[:8]}  last8={ids[-8:]}"
)
