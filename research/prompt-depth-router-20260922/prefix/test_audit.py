import csv
from pathlib import Path
import tempfile
import unittest

from audit import audit_prefix, coverage, inspected_prefix
from report import comparisons


def tsv(path, fields, records):
    with path.open("w") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(fields)
        writer.writerows(records)


class PrefixAuditTests(unittest.TestCase):
    def test_reconstructed_prefix_rejects_a_tail_token_and_byte_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = [1, 10, 11, 2]
            route = {"budget": "64", "tokens_read": "2", "user_tokens": "2",
                     "decoded_bytes": "8", "inspected_bytes": "8", "kind": "unknown",
                     "k": "3", "core_ns": "50", "routing_ns": "90"}
            tsv(root / "turn-1.prefix-span.tsv", ["start", "end", "leading_skip"], [[1, 3, 0]])
            (root / "turn-1.prefix.ids").write_text("10\n11\n")
            tsv(root / "turn-1.prefix-bytes.tsv", ["id", "hex"],
                [[10, b"Hello".hex()], [11, b" !\n".hex()]])
            (root / "turn-1.prefix.bin").write_bytes(b"Hello !\n")
            audit_prefix(root, 1, route, prompt)
            (root / "turn-1.prefix.ids").write_text("11\n2\n")
            with self.assertRaisesRegex(ValueError, "first user tokens"):
                audit_prefix(root, 1, route, prompt)
            (root / "turn-1.prefix.ids").write_text("10\n11\n")
            (root / "turn-1.prefix.bin").write_bytes(b"changed!")
            with self.assertRaisesRegex(ValueError, "reconstruct"):
                audit_prefix(root, 1, route, prompt)

    def test_utf8_and_word_boundary_do_not_require_the_next_token(self):
        self.assertEqual(inspected_prefix(b"Explain the design. \xf0", 64, 64), b"Explain the design. ")
        self.assertEqual(inspected_prefix(b"Write code", 64, 64), b"Write")
        self.assertEqual(inspected_prefix(b"Write code", 2, 64), b"Write code")
        self.assertEqual(inspected_prefix(b"\xff", 64, 64), b"")

    def test_code_coverage_requires_actual_parseable_closed_code(self):
        examples = [
            ("We could put Python between ``` markers later.", False),
            ("```python\ndef broken(:\n    pass\n```", False),
            ("```python\ndef validate_records(records):\n"
             "    \"\"\"Return a checked copy of a sequence of records.\"\"\"\n"
             "    if not isinstance(records, list):\n"
             "        raise TypeError('records must be a list')\n"
             "    return list(records)\n```", True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for text, expected in examples:
                (root / "turn-1.output.ids").write_text("7\n")
                (root / "turn-1.answer.txt").write_text(text)
                tsv(root / "token-bytes.tsv", ["id", "hex"], [[7, text.encode().hex()]])
                self.assertEqual(coverage(root, 1, "code")["code_covered"], expected)

    def test_complete_request_weighting_and_output_length_are_visible(self):
        def record(tokens, seconds):
            return {"output_tokens": tokens, "elapsed_s": seconds, "routing_ns": 100,
                    "k": 3, "prediction": "unknown",
                    "format": {"requested_format_covered": True,
                               "final_tokens_with_bytes": tokens, "nonfinal_tokens_with_bytes": 0}}
        matched = []
        for base, adapted in [((100, 1), (100, 2)), ((900, 9), (1800, 9))]:
            matched.append({"fixed3": record(*base), **{
                label: record(*adapted) for label in ("prefix64", "prefix128", "prefix256")}})
        result = comparisons(matched)["arms"]
        self.assertEqual(result["fixed3"]["tokens_per_second"], 100)
        self.assertAlmostEqual(result["prefix64"]["tokens_per_second"], 1900 / 11)
        self.assertEqual(result["prefix64"]["output_token_ratio_to_fixed3"], 1.9)
        self.assertEqual(result["prefix64"]["paired_gain_percent"], [-50, 100])


if __name__ == "__main__":
    unittest.main()
