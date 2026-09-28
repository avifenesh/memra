#!/usr/bin/env python3
"""Generate pinned Hugging Face tokenizers ID rows for raw text prompts."""

import hashlib
from pathlib import Path
import sys

import tokenizers
from tokenizers import Tokenizer


HERE = Path(__file__).resolve().parent
EXPECTED = {
    "tokenizer.json": "ff15eb925890d6b71b5160de4b846fbd13178438ab463b38ecc953e8cd1dcb3e",
    "tokenizer_config.json": "413a7845f52943ccf4de0e5c838414507d16c44dbf573da9e20bc8902b384d06",
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <pinned-hf-tokenizer-dir>")
    source = Path(sys.argv[1])
    for name, expected in EXPECTED.items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"{name}: SHA-256 mismatch: {actual}")

    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    lines = []
    for row in (HERE / "corpus.tsv").read_text().splitlines():
        name, hex_bytes = row.split("\t")
        text = bytes.fromhex(hex_bytes).decode("utf-8")
        special = tokenizer.encode(text, add_special_tokens=True).ids
        plain = tokenizer.encode(text, add_special_tokens=False).ids
        lines.append(
            f"{name}\t{','.join(map(str, special))}\t{','.join(map(str, plain))}\n"
        )
    (HERE / "ref-ids.tsv").write_text("".join(lines))
    print(f"tokenizers {tokenizers.__version__}: wrote {len(lines)} pinned cases")


if __name__ == "__main__":
    main()
