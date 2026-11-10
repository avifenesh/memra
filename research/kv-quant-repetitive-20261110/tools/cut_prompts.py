#!/usr/bin/env python3
"""Cut real documents into prompts of an exact token count (memra#1177).

Each prompt is the start of the document itself, raw (`--head raw`, the default), or a user
turn that opens with the chat template, "<|im_start|>user\n" + the document (`--head user`);
either way it ends inside the document. The cut is on a token
boundary: the first N ids of the encoded turn are decoded, and the decoded text must encode
back to exactly those N ids. If a cut splits a merge, the next shorter count that round-trips
is used, and the manifest records the exact count.

usage: cut_prompts.py TOKENIZER_JSON OUT_DIR NAME=PATH[:crlf] ... [--head raw|user] --tokens N [N ...]
`:crlf` turns CRLF line ends into LF before the cut (the Gutenberg text ships CRLF).
"""
import hashlib
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

HEADS = {"raw": "", "user": "<|im_start|>user\n"}


def main() -> None:
    args = sys.argv[1:]
    tok = Tokenizer.from_file(args[0])
    out = Path(args[1])
    out.mkdir(parents=True, exist_ok=True)
    head = "raw"
    if "--head" in args:
        h = args.index("--head")
        head = args[h + 1]
        del args[h : h + 2]
    HEAD = HEADS[head]
    k = args.index("--tokens")
    docs = args[2:k]
    targets = [int(x) for x in args[k + 1 :]]
    rows = []
    for spec in docs:
        name, _, rest = spec.partition("=")
        path, _, mode = rest.partition(":")
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        if mode == "crlf":
            text = text.replace("\r\n", "\n")
        # Enough characters for the largest target (digits and punctuation run near 2 chars
        # per token; prose near 4). Re-checked below.
        budget = max(targets) * 6
        body = text[:budget]
        ids = tok.encode(HEAD + body, add_special_tokens=False).ids
        for n in targets:
            if len(ids) < n + 64:
                raise SystemExit(f"{name}: only {len(ids)} tokens in the budget, need {n}")
            m = n
            while True:
                s = tok.decode(ids[:m], skip_special_tokens=False)
                back = tok.encode(s, add_special_tokens=False).ids
                if back == ids[:m]:
                    break
                m -= 1
            f = out / f"{name}-{m}.txt"
            f.write_text(s, encoding="utf-8")
            row = {
                "doc": name,
                "head": head,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "crlf_to_lf": mode == "crlf",
                "target": n,
                "tokens": m,
                "chars": len(s),
                "file": f.name,
                "prompt_sha256": hashlib.sha256(s.encode("utf-8")).hexdigest(),
                "ids_sha256": hashlib.sha256(json.dumps(ids[:m]).encode()).hexdigest(),
            }
            rows.append(row)
            print(json.dumps(row))
    with open(out / "manifest.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
