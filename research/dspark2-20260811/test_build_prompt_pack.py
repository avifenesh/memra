#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("build-prompt-pack.py")
SPEC = importlib.util.spec_from_file_location("build_prompt_pack", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildPromptPackTest(unittest.TestCase):
    def test_mode_and_split_are_exactly_cell_stratified(self) -> None:
        rows = []
        for category, count in {"chat": 8, "math": 12, "code": 12, "if": 8}.items():
            rows.extend(
                {"prompt": f"{category} prompt {index}", "category": category, "source": "test"}
                for index in range(count)
            )

        assigned, cells = MODULE.assign_mode_and_split(rows)
        self.assertEqual(sum(row["split"] == "heldout" for row in assigned), 2)
        self.assertEqual(sum(row["mode"] == "think" for row in assigned), 20)
        self.assertEqual(sum(row["mode"] == "nothink" for row in assigned), 20)
        self.assertEqual(sum(cells.values()), 40)
        for category in MODULE.CATEGORY_ORDER:
            category_rows = [row for row in assigned if row["category"] == category]
            self.assertEqual(
                sum(row["mode"] == "think" for row in category_rows),
                len(category_rows) // 2,
            )
            self.assertEqual(
                sum(row["mode"] == "nothink" for row in category_rows),
                len(category_rows) // 2,
            )

    def test_assignment_is_order_independent_per_prompt(self) -> None:
        rows = [
            {"prompt": f"prompt {index}", "category": "chat", "source": "test"}
            for index in range(40)
        ]
        forward, _ = MODULE.assign_mode_and_split(rows)
        reverse, _ = MODULE.assign_mode_and_split(list(reversed(rows)))
        by_prompt = lambda values: {
            row["prompt"]: (row["mode"], row["split"]) for row in values
        }
        self.assertEqual(by_prompt(forward), by_prompt(reverse))


if __name__ == "__main__":
    unittest.main()
