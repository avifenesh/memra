"""Capture HF-only consumed-whitespace expectations for the offline Rust tests."""
import json
from pathlib import Path

import tokenizers
from tokenizers import AddedToken, Tokenizer, normalizers

assert tokenizers.__version__ == "0.22.2"
root = Path(__file__).resolve().parents[2]
fixtures = root / "crates/memra-tokenizer/tests/fixtures/nfc"
base = (fixtures / "tiny-base-tokenizer.json").read_text()
variants = []
for nfc in (False, True):
    for normalized in (False, True):
        for name, tokens, inputs in (
            ("space-both", [(" ", True, True)], [" ", "  ", "     "]),
            ("pair-both", [("  ", True, True)], ["  ", "    ", "      "]),
            ("consumed-tab", [("<X>", False, True), ("\t", True, False)], ["<X>", "<X>\t"]),
            # HF deliberately keeps these nonempty raw matches. They prevent an overly
            # broad fix that discards everything whose original start precedes cursor.
            ("space-rstrip", [(" ", False, True)], [" ", "  ", "     "]),
        ):
            tokenizer = Tokenizer.from_str(base)
            tokenizer.normalizer = normalizers.NFC() if nfc else None
            for text, lstrip, rstrip in tokens:
                tokenizer.add_tokens([AddedToken(text, normalized=normalized, lstrip=lstrip, rstrip=rstrip)])
            source = json.loads(tokenizer.to_str())
            cases = []
            for text in inputs:
                expected = []
                for parse_special in (False, True):
                    tokenizer.encode_special_tokens = not parse_special
                    for add_special in (False, True):
                        expected.append({"parse_special": parse_special, "add_special": add_special,
                                         "ids": tokenizer.encode(text, add_special_tokens=add_special).ids})
                cases.append({"text": text, "expected": expected})
            variants.append({"name": f"{name}-nfc{int(nfc)}-normalized{int(normalized)}",
                             "normalizer": source["normalizer"], "added_tokens": source["added_tokens"],
                             "cases": cases})
# Load the exact declaration order instead of reserializing it: normalizer collisions
# keep file order within each class, even when token IDs are not ascending.
for reverse in (False, True):
    source = json.loads(base)
    source["model"]["vocab"]["e\u0301"] = 520
    source["normalizer"] = {"type": "NFC"}
    additions = [{"id": token_id, "content": text, "special": False, "normalized": True,
                  "single_word": False, "lstrip": False, "rstrip": False}
                 for token_id, text in [(520, "e\u0301"), (233, "é")]]
    source["added_tokens"] += additions[::-1] if reverse else additions
    tokenizer = Tokenizer.from_str(json.dumps(source))
    cases = []
    for text in ("é", "e\u0301"):
        expected = []
        for parse_special in (False, True):
            tokenizer.encode_special_tokens = not parse_special
            for add_special in (False, True):
                expected.append({"parse_special": parse_special, "add_special": add_special,
                                 "ids": tokenizer.encode(text, add_special_tokens=add_special).ids})
        cases.append({"text": text, "expected": expected})
    variants.append({"name": f"normalized-collision-order-{int(reverse)}",
                     "normalizer": source["normalizer"], "added_tokens": source["added_tokens"],
                     "vocab_additions": {"e\u0301": 520}, "cases": cases})
output = {"oracle": "Hugging Face tokenizers 0.22.2", "variants": variants}
(fixtures / "whitespace-oracle.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(f"Captured {len(variants)} variants, {sum(len(v['cases']) for v in variants)} inputs, four modes each")
