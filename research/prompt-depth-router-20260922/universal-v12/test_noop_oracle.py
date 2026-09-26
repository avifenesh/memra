"""A fixed-K C/D no-op must match its own K control, not K20."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import score_final
import score_validation


class NoopOracleTest(unittest.TestCase):
    def test_k3_noop_matches_k3_when_k20_output_differs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for phase, domain in (
                ("validation", "code"), ("final", "prose"),
            ):
                for label, value in (
                    ("fixed-k20-d3-c0", "20\n"),
                    ("fixed-k3-d3-c0", "3\n"),
                    ("cd-noop-fresh-k3", "3\n"),
                ):
                    session = root / f"{phase}-{domain}-0-{label}"
                    session.mkdir()
                    for turn in range(1, 9):
                        (session / f"turn-{turn}.output.ids").write_text(
                            value
                        )
            reference = {
                "cd-noop-fresh-k3": "fixed-k3-d3-c0",
            }
            with mock.patch.object(score_validation, "COUNT", 1):
                score_validation.identity(
                    root, "code", reference,
                )
                with self.assertRaises(ValueError):
                    score_validation.identity(
                        root, "code", {
                            "cd-noop-fresh-k3": "fixed-k20-d3-c0",
                        },
                    )
            with mock.patch.object(score_final, "COUNT", 1):
                score_final.noops_identical(
                    root, "prose", reference,
                )
                with self.assertRaises(ValueError):
                    score_final.noops_identical(
                        root, "prose", {
                            "cd-noop-fresh-k3": "fixed-k20-d3-c0",
                        },
                    )


if __name__ == "__main__":
    unittest.main()
