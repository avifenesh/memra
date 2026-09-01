#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from perf_acceptance_baseline import resolve_baseline


class AcceptanceBaselineTest(unittest.TestCase):
    def test_settled_baseline_must_match_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evidence.json").write_text(
                json.dumps({"acceptance_both_arms": 0.646}), encoding="utf-8"
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "id": "settled",
                                "acceptance_baseline": {
                                    "value": 0.646,
                                    "evidence": "evidence.json",
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_baseline(manifest, root / "missing.jsonl", "settled", root),
                0.646,
            )

            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["cells"][0]["acceptance_baseline"]["value"] = 0.816
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_baseline(manifest, root / "missing.jsonl", "settled", root)

    def test_history_fallback_uses_last_five_acceptance_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"cells": [{"id": "rolling"}]}), encoding="utf-8"
            )
            history = root / "history.jsonl"
            history.write_text(
                "\n".join(
                    json.dumps({"cell": "rolling", "accept": value})
                    for value in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_baseline(manifest, history, "rolling", root), 0.6
            )


if __name__ == "__main__":
    unittest.main()
