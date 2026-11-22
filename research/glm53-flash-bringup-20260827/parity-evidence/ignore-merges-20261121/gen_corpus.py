"""GLM-5.3-Flash: the texts on which tokenizer.json's `model.ignore_merges` decides the ids.

zai-org/GLM-5.3-Flash @ 04c4e9e95c5da8862dced7e5056455116f83a7e0 declares
`"ignore_merges": true`: a pre-split word that is itself a vocabulary piece encodes as that one
id, whatever the merge table would build. This script finds every piece for which that matters
(the tokenizers library with the flag as shipped against the same file with the flag off) and
writes each alone and inside a sentence, with the reference ids, in the tok-parity format.

    uv run --no-project --with tokenizers==0.23.3 python3 -I gen_corpus.py <dir with tokenizer.json>

Writes corpus.tsv and ref-ids.tsv next to this file. Refuses any other tokenizer.json.
"""

import hashlib
import json
import pathlib
import sys

import tokenizers
from tokenizers import Tokenizer

HERE = pathlib.Path(__file__).resolve().parent
PIN = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
path = pathlib.Path(sys.argv[1]) / "tokenizer.json"
blob = path.read_bytes()
assert hashlib.sha256(blob).hexdigest() == PIN, "tokenizer.json is not the pinned revision's"
assert tokenizers.__version__ == "0.23.3", tokenizers.__version__

raw = json.loads(blob)
assert raw["model"]["ignore_merges"] is True
shipped = Tokenizer.from_str(json.dumps(raw))
raw["model"]["ignore_merges"] = False
merges_only = Tokenizer.from_str(json.dumps(raw))

pieces = [shipped.decode([i]) for i in range(len(raw["model"]["vocab"]))]
with_flag = shipped.encode_batch(pieces, add_special_tokens=False)
without = merges_only.encode_batch(pieces, add_special_tokens=False)
decided = [i for i, (a, b) in enumerate(zip(with_flag, without)) if a.ids != b.ids]
assert all(with_flag[i].ids == [i] for i in decided)

corpus = []
for i in decided:
    corpus.append((f"piece-{i}", pieces[i]))
    corpus.append((f"sentence-{i}", f"GLM 说{pieces[i]}，对吗？ Yes{pieces[i]} 123."))
differ = sum(
    shipped.encode(text, add_special_tokens=False).ids != merges_only.encode(text, add_special_tokens=False).ids
    for _, text in corpus
)
with open(HERE / "corpus.tsv", "w") as f:
    for name, text in corpus:
        f.write(f"{name}\t{text.encode().hex()}\n")
with open(HERE / "ref-ids.tsv", "w") as f:
    for name, text in corpus:
        a = shipped.encode(text, add_special_tokens=True).ids
        b = shipped.encode(text, add_special_tokens=False).ids
        f.write(f"{name}\t{','.join(map(str, a))}\t{','.join(map(str, b))}\n")
shape = sum(p.startswith(" ") and len(p) > 1 for p in (pieces[i] for i in decided))
print(
    f"tokenizers {tokenizers.__version__}: pieces the flag decides {len(decided)} "
    f"(a space and a word: {shape}); corpus {len(corpus)}; texts whose ids move with the flag {differ}"
)
