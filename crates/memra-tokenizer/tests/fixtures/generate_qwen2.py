"""Generate synthetic BPE goldens with an independent, pinned HF tokenizer.

Run from the repo root:
uv run --no-project --with tokenizers==0.22.2 python crates/memra-tokenizer/tests/fixtures/generate_qwen2.py

No normalization or model weights are involved. The merge chains intentionally cross
potential regex boundaries, so an incorrect split changes observable token IDs.
"""
import json
from pathlib import Path

import tokenizers
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

assert tokenizers.__version__ == "0.22.2"
OUT = Path(__file__).parent
QWEN2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN35 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
TEXTS = [
    "", "e\u0301", "cafe\u0301", "x\u0301y", "a\u0301b\u0301c\u0301",
    "\u0301abc", "\u0301\u0308", "e\u0301  cafe\u0301", "café",
    "مُحَمَّد", "שָׁלוֹם", "नमस्ते", "ภาษาไทย", "中文かなカナ",
    "hello world", "  hello", "hello  ", "a\u00a0\u00a0b", "a\t\tb",
    "a \r\n\n b", "a\u2028b", "\t\n\r\n  ", "!?\u0301\r\n",
    "1234567", "١٢٣٤", "①②③④", "ⅨⅩⅪⅫ", "a12b34", "1\u03012",
    "We're I'M they'd", "We'Ve a'lL", "'ſx", "'ſe", "'Kx", "'\u0301x",
    "👩‍💻🙂", " a\x00b", "\x01\u0301abc", "a\u0301!\u0308b",
]

# GPT-2 byte-to-Unicode alphabet, independently constructed from its published rule.
kept = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
alphabet = {byte: chr(byte) for byte in kept}
for offset, byte in enumerate(byte for byte in range(256) if byte not in alphabet):
    alphabet[byte] = chr(256 + offset)
vocab = {alphabet[byte]: byte for byte in range(256)}
merges = []
for text in TEXTS:
    encoded = "".join(alphabet[byte] for byte in text.encode())
    if not encoded:
        continue
    prefix = encoded[0]
    for char in encoded[1:]:
        combined = prefix + char
        if combined not in vocab:
            vocab[combined] = len(vocab)
            merges.append((prefix, char))
        prefix = combined
eos = len(vocab)
vocab["<|endoftext|>"] = eos

def tokenizer(pattern):
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken("<|endoftext|>", special=True)])
    return tok

qwen2 = tokenizer(QWEN2)
qwen35 = tokenizer(QWEN35)
qwen2.save(str(OUT / "qwen2-tokenizer.json"), pretty=True)
oracle = {
    "oracle": "Hugging Face tokenizers 0.22.2; synthetic byte-level BPE; no normalizer",
    "eos_token_id": eos,
    "cases": [{"text": text, "ids": qwen2.encode(text, add_special_tokens=False).ids}
              for text in TEXTS],
    "qwen35_combining_ids": qwen35.encode("e\u0301", add_special_tokens=False).ids,
}
assert oracle["cases"][1]["ids"] != oracle["qwen35_combining_ids"]
(OUT / "qwen2-oracle.json").write_text(json.dumps(oracle, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {len(TEXTS)} cases, {len(vocab)} tokens and {len(merges)} merges")
