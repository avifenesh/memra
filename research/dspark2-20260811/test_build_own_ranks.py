#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("build-own-ranks.py")
SPEC = importlib.util.spec_from_file_location("build_own_ranks", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildOwnRanksTest(unittest.TestCase):
    def test_counts_response_only_and_backfills_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pairs = Path(directory) / "pairs.tsv"
            pairs.write_text(
                "# memra-dspark-pairs-v1\n"
                "10\ttrain\tthink\tcode\t2\t4\t6\t100 248044 7 7 9 248046\n"
                "11\theldout\tnothink\tchat\t1\t3\t4\t248045 8 7 8\n"
            )
            frequency, split_frequency, high_ids, stats = MODULE.parse_pairs([pairs])
            self.assertEqual(frequency, {7: 2, 9: 1, 248046: 1})
            self.assertNotIn(100, frequency)
            self.assertEqual(stats["pairs"], 2)
            self.assertEqual(stats["response_tokens"], 7)
            self.assertEqual(stats["ranking_split"], "train")
            self.assertEqual(stats["ranking_response_tokens"], 4)
            self.assertEqual(stats["ranking_distinct_response_ids"], 3)
            self.assertEqual(high_ids, {248046})

            selected, backfill_used = MODULE.select_ids(
                frequency, high_ids, [7, 9, 10, 11, 12, 13], 10
            )
            self.assertEqual(selected[:5], list(MODULE.FROZEN_SPECIAL_IDS))
            self.assertEqual(selected[5:7], [7, 9])
            self.assertEqual(selected[7:], [10, 11, 12])
            self.assertEqual(backfill_used, 3)
            self.assertEqual(MODULE.coverage(split_frequency["train"], set(selected)), 1.0)
            self.assertEqual(
                MODULE.coverage(split_frequency["heldout"], set(selected)), 1.0 / 3.0
            )

    def test_duplicate_pair_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.tsv"
            second = Path(directory) / "second.tsv"
            row = "3\ttrain\tthink\tcode\t1\t1\t2\t1 2\n"
            first.write_text("# memra-dspark-pairs-v1\n" + row)
            second.write_text("# memra-dspark-pairs-v1\n" + row)
            with self.assertRaisesRegex(ValueError, "duplicate pair id 3"):
                MODULE.parse_pairs([first, second])


if __name__ == "__main__":
    unittest.main()
