"""Is the whole-word lookup (tokenizer.json `model.ignore_merges`) the o200k family's own program?

The artifact also ships its vocabulary in tiktoken's format (`tiktoken/tokenizer.model`, the
o200k_base layout: base64 piece, rank). tiktoken looks a whole regex piece up before it merges
byte pairs. This script builds a tiktoken Encoding from that file, the artifact's Split regex and
its added tokens, and encodes the corpus of gen_fixtures.py three ways.

    uv run --no-project --with tiktoken==0.12.0 --with tokenizers==0.23.3 \
        python3 -I verify_tiktoken.py <dir with tiktoken/tokenizer.model>

Prints one line a comparison; exits 1 unless tiktoken equals tokenizers with the flag as shipped
on every case and differs from tokenizers with the flag off on exactly the `unmerged-*` cases.
"""

import base64
import hashlib
import json
import pathlib
import sys

import tiktoken
from tokenizers import Tokenizer

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE / "artifact"
model_path = pathlib.Path(sys.argv[1]) / "tiktoken" / "tokenizer.model"
blob = model_path.read_bytes()
print(f"tiktoken {tiktoken.__version__} tokenizer.model sha256 {hashlib.sha256(blob).hexdigest()}")

raw = json.loads((ART / "tokenizer.json").read_text())
regex = raw["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
ranks = {}
for line in blob.splitlines():
    piece, rank = line.split()
    ranks[base64.b64decode(piece)] = int(rank)
special = {t["content"]: t["id"] for t in raw["added_tokens"]}
base = {piece: rank for piece, rank in ranks.items() if rank < 199998}
print(f"pieces {len(ranks)} below the added tokens {len(base)}")
enc = tiktoken.Encoding("inkling2", pat_str=regex, mergeable_ranks=base, special_tokens=special)

shipped = Tokenizer.from_file(str(ART / "tokenizer.json"))
assert raw["model"]["ignore_merges"] is True
raw["model"]["ignore_merges"] = False
merges_only = Tokenizer.from_str(json.dumps(raw))

cases = []
for line in (HERE / "corpus.tsv").read_text().splitlines():
    name, hexed = line.split("\t")
    cases.append((name, bytes.fromhex(hexed).decode()))
skipped, vs_shipped, vs_off = [], [], []
for name, text in cases:
    try:
        ids = enc.encode(text, allowed_special="all")
    except Exception as error:  # tiktoken's regex engine refuses a few inputs
        skipped.append((name, type(error).__name__))
        continue
    if ids != shipped.encode(text, add_special_tokens=False).ids:
        vs_shipped.append(name)
    if ids != merges_only.encode(text, add_special_tokens=False).ids:
        vs_off.append(name)
print(f"cases {len(cases)} skipped {len(skipped)} {skipped[:5]}")
print(f"tiktoken != tokenizers(ignore_merges=true): {len(vs_shipped)} {vs_shipped[:10]}")
unmerged = [n for n in vs_off if n.startswith('unmerged-')]
print(f"tiktoken != tokenizers(ignore_merges=false): {len(vs_off)}, of them unmerged-*: {len(unmerged)}")
ok = not vs_shipped and len(vs_off) == len(unmerged) == 337
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
