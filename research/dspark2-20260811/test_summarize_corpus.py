#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("summarize-corpus.py")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class SummarizeCorpusTest(unittest.TestCase):
    def test_reports_assignment_scoped_anchor_deficiencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "chunks"
            chunk = root / "pilot-00000-00002"
            write(
                chunk / "generated" / "pairs.tsv",
                "# memra-dspark-pairs-v1\n"
                "0\ttrain\tnothink\tchat\t2\t5\t7\t1 2 3 4 5 6 7\n"
                "1\theldout\tthink\tmath\t2\t2\t4\t8 9 10 11\n",
            )
            write(
                chunk / "extracted" / "extraction.meta.json",
                json.dumps(
                    {
                        "records": 3,
                        "anchors_per_pair": 4,
                        "skipped_short": 1,
                    }
                ),
            )
            write(
                chunk / "extracted" / "index.tsv",
                "record\tpair_id\tanchor_pos\tprompt_len\tsplit\tmode\tcategory\n"
                "0\t0\t1\t2\ttrain\tnothink\tchat\n"
                "1\t0\t2\t2\ttrain\tnothink\tchat\n"
                "2\t0\t3\t2\ttrain\tnothink\tchat\n",
            )
            write(
                chunk / "validation.json",
                json.dumps(
                    {
                        "records": 3,
                        "sampled_token_top64_rate": 1.0,
                        "tail_mass_mean": 0.1,
                        "tail_mass_max": 0.2,
                        "max_probability_mass_error": 1.0e-8,
                        "max_logit_probability_ratio_error": 2.0e-7,
                    }
                ),
            )
            manifest_paths = [
                "generated/pairs.tsv",
                "extracted/extraction.meta.json",
                "extracted/index.tsv",
                "validation.json",
            ]
            manifest = "".join(
                f"{hashlib.sha256((chunk / relative).read_bytes()).hexdigest()}  {relative}\n"
                for relative in manifest_paths
            )
            write(chunk / "sha256.txt", manifest)
            (chunk / ".remote-verified").touch()

            output = Path(temporary) / "summary.json"
            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--root",
                    str(root),
                    "--label",
                    "pilot",
                    "--start",
                    "0",
                    "--end",
                    "2",
                    "--out",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(output.read_text())
            self.assertEqual(
                summary["assignment_cells"],
                {"chat/nothink/train": 1, "math/think/heldout": 1},
            )
            self.assertEqual(summary["anchor_sampling"]["actual_records"], 3)
            self.assertEqual(summary["anchor_sampling"]["deficient_records"], 5)
            self.assertEqual(
                summary["anchor_sampling"]["deficient_records_by_cell"],
                {"chat/nothink/train": 1, "math/think/heldout": 4},
            )
            self.assertEqual(
                summary["anchor_sampling"]["zero_anchor_pairs_by_cell"],
                {"math/think/heldout": 1},
            )


if __name__ == "__main__":
    unittest.main()
