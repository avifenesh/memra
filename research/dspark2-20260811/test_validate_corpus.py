#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("validate-corpus.py")
SPEC = importlib.util.spec_from_file_location("validate_corpus", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ValidateCorpusTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> None:
        records, hidden, gamma, top_k = 1, 4096, 5, 64
        (root / "extraction.meta.json").write_text(
            json.dumps(
                {
                    "format": "memra-dspark-anchors-v1",
                    "pairs": 1,
                    "records": records,
                    "skipped_short": 0,
                    "hidden_size": hidden,
                    "anchors_per_pair": 4,
                    "gamma": gamma,
                    "top_k": top_k,
                    "temperature": 0.7,
                    "chunk": 512,
                    "seed": 20260811,
                }
            )
        )
        np.zeros(records * hidden, dtype="<u2").tofile(root / "hiddens.bf16")
        np.arange(gamma + 1, dtype="<u4").tofile(root / "tokens.u32")
        np.tile(np.arange(top_k, dtype="<u4"), gamma).tofile(root / "top_ids.u32")
        row_logits = np.linspace(3.0, -3.0, top_k, dtype="<f4")
        logits = np.tile(row_logits, gamma)
        logits.tofile(root / "top_logits.f32")
        numerators = np.exp(row_logits.astype(np.float64) / 0.7)
        row_probs = (0.8 * numerators / numerators.sum()).astype("<f4")
        np.tile(row_probs, gamma).tofile(root / "top_probs.f32")
        np.full(gamma, 1.0 - row_probs.sum(dtype=np.float64), dtype="<f4").tofile(
            root / "tail_probs.f32"
        )
        (root / "index.tsv").write_text(
            "record\tpair_id\tanchor_pos\tprompt_len\tsplit\tmode\tcategory\n"
            "0\t3\t10\t8\ttrain\tthink\tcode\n"
        )

    def test_valid_frozen_shape_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            result = MODULE.validate(root)
            self.assertEqual(result["records"], 1)
            self.assertLess(result["max_probability_mass_error"], 2.0e-5)

    def test_truncated_column_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            (root / "tokens.u32").write_bytes(b"\0" * 4)
            with self.assertRaisesRegex(ValueError, "bytes, expected"):
                MODULE.validate(root)


if __name__ == "__main__":
    unittest.main()
